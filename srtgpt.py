import re
import argparse
import asyncio
import json
import logging
from asyncio import Semaphore
from datetime import datetime
from pathlib import Path
from typing import List, NamedTuple, Tuple

import httpx


def parse_timestamp(timestamp: str) -> int:
    """Convert SRT timestamp to milliseconds for comparison."""
    hours, minutes, seconds = timestamp.replace(',', '.').split(':')
    return int(float(hours) * 3600000 + float(minutes) * 60000 + float(seconds) * 1000)


def setup_logger(level=logging.INFO, name=__name__):
    # Set root logger to WARNING to suppress other libraries' logs
    logging.getLogger().setLevel(logging.WARNING)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create handler
    handler = logging.StreamHandler()

    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Add formatter to handler
    handler.setFormatter(formatter)

    # Add handler to logger if it doesn't already have one
    if not logger.handlers:
        logger.addHandler(handler)

    return logger


class TranslationError(Exception): ...


class TooManyBlocksError(Exception): ...


class SubtitleBlock(NamedTuple):
    number: int
    timestamp: str
    text: str


class TranslationResult(NamedTuple):
    number: int
    timestamp: str
    text: str
    original_text: str


class TranAPI:
    def __init__(self, api_base: str, api_key: str, model: str = "gemini-2.5-flash", max_retries: int = 3, timeout: float = 60.0, initial_delay: float = 1.0):
        self.api_endpoint = f"{api_base}/v1/chat/completions"
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.client = httpx.AsyncClient(timeout=timeout)

    def create_prompt(self, texts: List[str]) -> str:
        return f"""Translate the following subtitles into Traditional Chinese (繁體中文). Ensure that the tone of your translation matches the context of the subtitles, and adapt idiomatic expressions to fit naturally into Traditional Chinese culture and language.

### Rules:
1. **Preserve Line Breaks**: Each line must correspond to a single Traditional Chinese line in the same array position. Do not combine or merge multiple lines.  
2. **Maintain Array Length**: The output array must contain the same number of items (lines) as the input array.  
3. **Contextual Translations**: Ensure the tone of the translation matches the context of the subtitle (e.g., casual, formal, emotional). Idiomatic expressions should be contextually adapted for naturalness rather than directly translated.  
4. **Subtitle-Friendly**: Keep translations concise, adhering to typical subtitle length limits. Ensure readability and clarity.  
5. **Strict Output Format**: Provide the output **ONLY** as a JSON array of strings without any additional comments, explanations, or formatting outside the JSON array. MUST be valid JSON that can be parsed as List[str].
6. **Name Translation**: All personal names must be translated into Chinese characters, maintaining appropriate cultural context and common Chinese naming conventions.

### Examples of Correct and Incorrect Formatting:

**Input Lines:**
["when I barely sell enough milkshakes", "to justify my single spindle? Right?"]

**Incorrect (DO NOT DO THIS):**
["當我目前的奶昔銷量甚至連單軸攪拌機的需求都撐不起來呢？對吧？"]

**Correct (DO THIS):**
["當我目前的奶昔銷量甚至連", "單軸攪拌機的需求都撐不起來呢？對吧？"]

**Input Lines:**
["- Back off. - I'm gonna."]

**Incorrect (DO NOT DO THIS):**
["- 滾開。", "- 我會的。"]

**Correct (DO THIS):**
["- 滾開。- 我會的。"]

### Your Task:
Translate the input array below and return only a JSON array of the corresponding Traditional Chinese translations.

**Input Lines:**
{json.dumps(texts, ensure_ascii=False)}
"""

    async def llm(self, content: str) -> dict:
        for i in range(self.max_retries):
            try:
                response = await self.client.post(
                    self.api_endpoint,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}"
                    },
                    json={
                        "model": self.model,
                        "stream": False,
                        "temperature": 0.2,
                        "top_p": 1,
                        "messages": [{"role": "user", "content": content}],
                    },
                )
                response.raise_for_status()

                result = response.json()
                return result["choices"][0]["message"]["content"]
            except Exception as e:
                delay_time = self.initial_delay * (2 ** i)
                logger.error(
                    f"Error in API: {str(e)}. Retrying in {delay_time} seconds..."
                )
                await asyncio.sleep(delay_time)

        raise RuntimeError(f"API failed after {self.max_retries} retries")

    async def translate(self, texts: List[str]) -> List[str]:
        for _ in range(2):
            try:
                prompt = self.create_prompt(texts)
                translated_text = await self.llm(prompt)

                start, end = translated_text.find("["), translated_text.rfind("]")
                if start == -1 or end == -1:
                    logger.debug(f"prompt: {prompt}")
                    logger.debug(f"response: {translated_text}")
                    raise TranslationError("No JSON array found in response")

                try:
                    translated_list = json.loads(translated_text[start : end + 1])
                except json.decoder.JSONDecodeError:
                    raise TranslationError("Not a valid JSON")

                if len(translated_list) != len(texts):
                    raise TranslationError(
                        f"Translation count mismatch: got {len(translated_list)}, expected {len(texts)}"
                    )

                if not all(isinstance(text, str) for text in translated_list):
                    raise TranslationError("Not all translations are strings")

                return translated_list

            except TranslationError as e:
                # might be quite often, only print in debug message
                logger.debug(f"Error in translation: {str(e)}. Retrying...")

        if len(texts) == 1:
            raise RuntimeError("Translation failed after many tries...")

        return await self.translate(texts[: len(texts) // 2]) + await self.translate(
            texts[len(texts) // 2 :]
        )


class TranSRT:
    def __init__(self, tran_api: TranAPI, max_concurrent_requests: int, filter_bad_words: bool = False):
        self.tran_api = tran_api
        self.semaphore = Semaphore(max_concurrent_requests)
        self.filter_bad_words = filter_bad_words

    @staticmethod
    def parse_file(file_path: str, filter_bad_words: bool = False) -> List[SubtitleBlock]:
        def normalize_spaces(text):
            return re.sub(r'\s+', ' ', text)

        def filter_bad(text):
            bad_words = {
                'dick': 'dic*',
                'pussy': 'pu**y',
                'blow job': 'blo* job',
                'rape': 'rap*',
                'orgasm': 'orgas*',
                'had sex': 'haɗ seҳ',
                'have sex': 'hav* seҳ',
                'masturbate': 'masturbat*',
            }

            for word, censored in bad_words.items():
                text = text.replace(word, censored)
            return text

        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        blocks = content.strip().split("\n\n")
        parsed_blocks = []
        last_timestamp = -1
        block_count = 0

        for block in blocks:
            lines = block.strip().split("\n")
            if len(lines) >= 3:
                # Check total number of blocks, some bad srt files have too many blocks
                block_count += 1
                if block_count > 5000:
                    raise TooManyBlocksError(f"SRT file has too many subtitle blocks ({block_count} > 5000), might be broken or malformed")

                timestamp = lines[1].split(" --> ")[0]  # Get start timestamp
                current_timestamp = parse_timestamp(timestamp)
                
                # Stop parsing if timestamp is out of order
                if current_timestamp < last_timestamp:
                    logger.info(f"Stopping at subtitle #{lines[0]} due to out-of-order timestamp")
                    break
                last_timestamp = current_timestamp

                text = " ".join(lines[2:])
                text = normalize_spaces(text)
                
                # Skip single character subtitles
                if len(text.strip()) <= 1:
                    logger.debug(f"Skipping single character subtitle #{lines[0]}: '{text}'")
                    continue
                
                if filter_bad_words:
                    text = filter_bad(text)

                parsed_blocks.append(
                    SubtitleBlock(
                        number=int(lines[0]),
                        timestamp=lines[1],
                        text=text
                    )
                )

        return parsed_blocks

    @staticmethod
    def write_file(output_file: str, blocks: List[TranslationResult]):
        with open(output_file, "w", encoding="utf-8") as file:
            for block in blocks:
                file.write(f"{block.number}\n")
                file.write(f"{block.timestamp}\n")
                file.write(f"{block.text}\n\n")

    async def translate_batch(
        self,
        batch: List[SubtitleBlock],
        batch_num: int,
        total_batches: int,
    ) -> List[TranslationResult]:
        async with self.semaphore:
            texts = [block.text for block in batch]
            translated_texts = await self.tran_api.translate(texts)

            results = [
                TranslationResult(
                    number=block.number,
                    timestamp=block.timestamp,
                    text=translated_text,
                    original_text=block.text,
                )
                for block, translated_text in zip(batch, translated_texts)
            ]

            logger.info(f"Batch {batch_num}/{total_batches}: Completed")
            return results

    async def translate_file(self, input_file: str, output_file: str, batch_size: int):
        # Parse input file
        blocks = self.parse_file(input_file, self.filter_bad_words)
        total_blocks = len(blocks)

        # Prepare batches
        batches = [
            blocks[i : i + batch_size] for i in range(0, total_blocks, batch_size)
        ]

        # Process all batches
        translated_blocks = []

        tasks = [
            self.translate_batch(batch, i + 1, len(batches))
            for i, batch in enumerate(batches)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                raise RuntimeError(f"Batch {i+1} failed: {str(result)}")
            else:
                translated_blocks.extend(result)

        # Sort and write results
        translated_blocks.sort(key=lambda x: x.number)
        self.write_file(output_file, translated_blocks)

        logger.info(
            f"Translation completed: {len(translated_blocks)}/{total_blocks} subtitles"
        )


def parse_srt_filename(input_path: str) -> Tuple[Path, str, str]:
    if not input_path.endswith(".srt"):
        raise ValueError(f"Not .srt file: {input_path}")

    name, _, lang = input_path[:-4].replace(".hi", "").rpartition(".")
    folder, _, name = name.rpartition("/")
    return Path(folder), name, lang


def parse_args():
    parser = argparse.ArgumentParser(
        description="Translate SRT subtitles to Traditional Chinese"
    )
    parser.add_argument("input_path", help="Input SRT file path or folder containing SRT files")
    parser.add_argument(
        "--api-base",
        help="OpenAI API proxy server base URL",
    )
    parser.add_argument(
        "--api-key",
        required=True,
        help="API key for authentication",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="LLM model",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of subtitles to translate in one batch",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retries for failed requests",
    )
    parser.add_argument(
        "--initial-delay",
        type=float,
        default=1.0,
        help="Initial delay time in seconds for retries (will be doubled each retry)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Maximum number of concurrent API requests",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Timeout in seconds for API requests",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force translation even if Chinese version exists",
    )
    parser.add_argument(
        "--filter-bad-words",
        action="store_true",
        help="Filter out bad words in subtitles",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )

    return parser.parse_args()


async def process_file(input_file: str, tran_srt: TranSRT, args):
    try:
        folder, name, lang = parse_srt_filename(input_file)

        # Skip if already Chinese
        if lang in ["zh-TW", "zh"]:
            logger.info(f"Already Chinese subtitles: {lang}, ignore...")
            return

        if not args.force:
            # check existing translated subtitles
            for lang in ["zh-TW"]:
                if (folder / f"{name}.{lang}.srt").exists():
                    logger.info(
                        f"Chinese translation already exists for {name}, use --force to translate anyway"
                    )
                    return

        try:
            await tran_srt.translate_file(
                input_file, str(folder / f"{name}.zh-TW.srt"), args.batch_size
            )
        except TooManyBlocksError as e:
            logger.error(f"Error processing {input_file}: {str(e)}")
            return False
    except Exception as e:
        logger.error(f"Error processing {input_file}: {str(e)}")
        return False
    return True


async def main():
    global logger

    args = parse_args()

    log_level = getattr(logging, args.log_level.upper())
    logger = setup_logger(level=log_level)

    input_path = Path(args.input_path)
    
    # Initialize translation API and SRT handler
    tran_api = TranAPI(
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        max_retries=args.max_retries,
        timeout=args.timeout,
        initial_delay=args.initial_delay,
    )
    tran_srt = TranSRT(
        tran_api=tran_api,
        max_concurrent_requests=args.max_concurrent,
        filter_bad_words=args.filter_bad_words,
    )

    if input_path.is_file():
        # Process single file
        if not input_path.name.endswith(".srt"):
            logger.error(f"Not .srt file: {input_path}")
            return
        await process_file(str(input_path), tran_srt, args)
    elif input_path.is_dir():
        # Process all .srt files in directory
        srt_files = list(input_path.glob("**/*.srt"))
        if not srt_files:
            logger.error(f"No .srt files found in {input_path}")
            return
            
        logger.info(f"Found {len(srt_files)} .srt files to process")
        for srt_file in srt_files:
            logger.info(f"Processing {srt_file}")
            await process_file(str(srt_file), tran_srt, args)
    else:
        logger.error(f"Input path does not exist: {input_path}")
        return


if __name__ == "__main__":
    asyncio.run(main())
