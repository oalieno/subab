import re
import argparse
import asyncio
import json
import logging
from asyncio import Semaphore
from pathlib import Path
from typing import List, NamedTuple, Tuple

import httpx  # type: ignore


def parse_timestamp(timestamp: str) -> int:
    """Convert SRT timestamp to milliseconds for comparison."""
    try:
        hours, minutes, seconds = timestamp.replace(',', '.').split(':')
        return int(float(hours) * 3600000 + float(minutes) * 60000 + float(seconds) * 1000)
    except (ValueError, AttributeError) as e:
        logger.warning(f"Failed to parse timestamp '{timestamp}': {e}")
        return -1


def ms_to_timestamp(ms: int) -> str:
    """Convert milliseconds to SRT timestamp format."""
    if ms < 0:
        ms = 0
    hours = ms // 3600000
    ms %= 3600000
    minutes = ms // 60000
    ms %= 60000
    seconds = ms // 1000
    milliseconds = ms % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


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


class RateLimitError(Exception):
    """Rate limit exceeded, should wait before retrying."""
    def __init__(self, message: str, retry_after: float = None):
        super().__init__(message)
        self.retry_after = retry_after


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
        self.timeout = timeout
        self._client = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()
        return False

    @property
    def client(self):
        if self._client is None:
            raise RuntimeError("TranAPI must be used as async context manager")
        return self._client

    def create_prompt(self, texts: List[str], target_language: str) -> str:
        return f"""Translate the following subtitles into {target_language}. Your main goal is to produce a natural and accurate translation while strictly preserving the original subtitle structure.

### Core Rules:
1.  **1-to-1 Line Mapping**: Each string in the input JSON array corresponds to one line of subtitle. Your output must be a JSON array with the exact same number of strings. Do not merge lines or split lines.
2.  **Contextual Accuracy**: Ensure the tone (e.g., casual, formal) and meaning match the original context. Adapt idioms naturally into {target_language}.
3.  **Conciseness**: Keep translations subtitle-friendly—clear and easy to read quickly.
4.  **Strict JSON Output**: The output MUST be a valid JSON array of strings. Do not include any explanations, markdown, or any text outside of the JSON array.
5.  **Name Translation**: All personal names must be translated or transliterated into {target_language}, maintaining appropriate cultural context and common naming conventions for that language.

### Structural Examples (How to handle line breaks):

These examples illustrate the structural rules. You will translate the content into {target_language}.

**Example 1: Preserving intended line breaks.**
*   **Input Lines**: `["when I barely sell enough milkshakes", "to justify my single spindle? Right?"]`
*   **Analysis**: The original subtitle was intentionally split into two lines, likely for timing or readability.
*   **Correct Output Structure**: The translation must also be a JSON array of two strings.
    `["<translation of first line>", "<translation of second line>"]`
*   **Incorrect Output Structure (DO NOT DO THIS)**: Merging the lines into one string.
    `["<translation of both lines combined into one>"]`

**Example 2: Keeping single-line dialogues intact.**
*   **Input Lines**: `["- Back off. - I'm gonna."]`
*   **Analysis**: This is a single line containing a quick exchange. It should remain a single line.
*   **Correct Output Structure**: The translation must be a JSON array with a single string.
    `["<translation of the full line>"]`
*   **Incorrect Output Structure (DO NOT DO THIS)**: Splitting one line into multiple strings in the array.
    `["<translation of '- Back off.'>", "<translation of '- I'm gonna.'>"]`

### Your Task:
Translate the following input array into {target_language}, strictly following all the rules above.

**Input Lines:**
{json.dumps(texts, ensure_ascii=False)}
"""

    async def llm(self, content: str) -> str:
        last_exception = None
        
        for attempt in range(self.max_retries):
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
                
                # Handle different HTTP status codes
                if response.status_code == 429:
                    # Rate limit - check for Retry-After header
                    retry_after = response.headers.get("Retry-After")
                    if retry_after:
                        try:
                            wait_time = float(retry_after)
                        except ValueError:
                            wait_time = self.initial_delay * (2 ** attempt)
                    else:
                        wait_time = self.initial_delay * (2 ** attempt)
                    
                    logger.warning(f"Rate limit exceeded (429). Waiting {wait_time:.1f}s before retry {attempt + 1}/{self.max_retries}")
                    await asyncio.sleep(wait_time)
                    last_exception = RateLimitError("Rate limit exceeded", wait_time)
                    continue
                
                elif 400 <= response.status_code < 500:
                    # Client errors (except 429) - don't retry
                    error_msg = "Client error {}: {}".format(response.status_code, response.text)
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                
                elif 500 <= response.status_code < 600:
                    # Server errors - retry with backoff
                    delay_time = self.initial_delay * (2 ** attempt)
                    logger.warning(f"Server error {response.status_code}. Retrying in {delay_time:.1f}s (attempt {attempt + 1}/{self.max_retries})")
                    await asyncio.sleep(delay_time)
                    last_exception = RuntimeError(f"Server error {response.status_code}")
                    continue
                
                # Successful response
                response.raise_for_status()
                
                # Parse and validate response
                try:
                    result = response.json()
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON response: {response.text}")
                    raise RuntimeError(f"Invalid JSON response from API: {e}")
                
                # Validate response structure
                if not isinstance(result, dict):
                    raise RuntimeError(f"Unexpected response type: {type(result)}")
                
                if "choices" not in result:
                    raise RuntimeError(f"Missing 'choices' in API response: {result}")
                
                if not result["choices"] or len(result["choices"]) == 0:
                    raise RuntimeError(f"Empty 'choices' in API response: {result}")
                
                if "message" not in result["choices"][0]:
                    raise RuntimeError(f"Missing 'message' in API response: {result}")
                
                if "content" not in result["choices"][0]["message"]:
                    raise RuntimeError(f"Missing 'content' in API response: {result}")
                
                return result["choices"][0]["message"]["content"]
                
            except httpx.TimeoutException as e:
                delay_time = self.initial_delay * (2 ** attempt)
                logger.warning(f"Request timeout. Retrying in {delay_time:.1f}s (attempt {attempt + 1}/{self.max_retries})")
                await asyncio.sleep(delay_time)
                last_exception = e
                
            except httpx.NetworkError as e:
                delay_time = self.initial_delay * (2 ** attempt)
                logger.warning(f"Network error: {str(e)}. Retrying in {delay_time:.1f}s (attempt {attempt + 1}/{self.max_retries})")
                await asyncio.sleep(delay_time)
                last_exception = e
                
            except RuntimeError:
                # Don't retry RuntimeError (client errors, validation errors)
                raise
                
            except Exception as e:
                delay_time = self.initial_delay * (2 ** attempt)
                logger.error(f"Unexpected error: {str(e)}. Retrying in {delay_time:.1f}s (attempt {attempt + 1}/{self.max_retries})")
                await asyncio.sleep(delay_time)
                last_exception = e

        raise RuntimeError(f"API failed after {self.max_retries} retries. Last error: {last_exception}")

    @staticmethod
    def clean_json_trailing_commas(json_str: str) -> str:
        """Remove trailing commas from JSON string that would cause parsing errors."""
        # Remove trailing commas before closing brackets/braces
        # Pattern 1: , followed by optional whitespace and ]
        json_str = re.sub(r',(\s*])', r'\1', json_str)
        # Pattern 2: , followed by optional whitespace and }
        json_str = re.sub(r',(\s*})', r'\1', json_str)
        return json_str

    async def translate(self, texts: List[str], target_language: str) -> List[str]:
        for _ in range(2):
            try:
                prompt = self.create_prompt(texts, target_language)
                translated_text = await self.llm(prompt)

                start, end = translated_text.find("["), translated_text.rfind("]")
                if start == -1 or end == -1:
                    logger.debug(f"prompt: {prompt}")
                    logger.debug(f"response: {translated_text}")
                    raise TranslationError("No JSON array found in response")

                try:
                    json_text = translated_text[start : end + 1]
                    # Clean trailing commas before parsing
                    json_text = self.clean_json_trailing_commas(json_text)
                    translated_list = json.loads(json_text)
                except json.decoder.JSONDecodeError as e:
                    logger.debug(f"JSON decode error: {str(e)}")
                    logger.debug(f"Problematic JSON: {translated_text[start : end + 1]}")
                    raise TranslationError("Not a valid JSON")

                if len(translated_list) != len(texts):
                    raise TranslationError(
                        f"Translation count mismatch: got {len(translated_list)}, expected {len(texts)}"
                    )

                if not all(isinstance(text, str) for text in translated_list):
                    raise TranslationError(f"Not all translations are strings: {translated_list}")

                return translated_list

            except TranslationError as e:
                # might be quite often, only print in debug message
                logger.debug(f"Error in translation: {str(e)}. Retrying...")

        if len(texts) == 1:
            raise RuntimeError("Translation failed after many tries...")

        return await self.translate(texts[: len(texts) // 2], target_language) + await self.translate(
            texts[len(texts) // 2 :], target_language
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
                # Case-insensitive replacement
                text = re.sub(re.escape(word), censored, text, flags=re.IGNORECASE)
            return text

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
        except UnicodeDecodeError:
            # Try with different encoding
            logger.warning(f"Failed to read {file_path} with UTF-8, trying with latin-1")
            with open(file_path, "r", encoding="latin-1") as file:
                content = file.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to read file {file_path}: {e}")

        blocks = content.strip().split("\n\n")
        parsed_blocks = []
        last_timestamp = -1
        block_count = 0

        for block_idx, block in enumerate(blocks):
            lines = block.strip().split("\n")
            if len(lines) >= 3:
                try:
                    # Check total number of blocks, some bad srt files have too many blocks
                    block_count += 1
                    if block_count > 5000:
                        raise TooManyBlocksError(f"SRT file has too many subtitle blocks ({block_count} > 5000), might be broken or malformed")

                    # Parse subtitle number
                    try:
                        subtitle_number = int(lines[0])
                    except ValueError:
                        logger.debug(f"Skipping block {block_idx}: invalid subtitle number '{lines[0]}'")
                        continue

                    # Parse timestamp
                    if " --> " not in lines[1]:
                        logger.debug(f"Skipping block {block_idx}: invalid timestamp format '{lines[1]}'")
                        continue
                    
                    timestamp_parts = lines[1].split(" --> ")
                    if len(timestamp_parts) != 2:
                        logger.debug(f"Skipping block {block_idx}: malformed timestamp '{lines[1]}'")
                        continue
                    
                    timestamp = timestamp_parts[0]
                    current_timestamp = parse_timestamp(timestamp)
                    
                    # Skip if timestamp parsing failed
                    if current_timestamp == -1:
                        logger.debug(f"Skipping block {block_idx}: failed to parse timestamp")
                        continue
                    
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
                            number=subtitle_number,
                            timestamp=lines[1],
                            text=text
                        )
                    )
                except TooManyBlocksError:
                    raise
                except Exception as e:
                    logger.warning(f"Error parsing block {block_idx}: {e}. Skipping...")
                    continue

        if not parsed_blocks:
            raise ValueError(f"No valid subtitle blocks found in {file_path}")

        return parsed_blocks

    @staticmethod
    def write_file(output_file: str, blocks: List[TranslationResult]):
        with open(output_file, "w", encoding="utf-8") as file:
            for block in blocks:
                file.write(f"{block.number}\n")
                file.write(f"{block.timestamp}\n")
                file.write(f"{block.text}\n\n")

    @staticmethod
    def post_process_translation(text: str) -> str:
        """Applies various post-processing cleanups to the translated text."""
        # Split dialogues like "- Hello - World" into separate lines for readability.
        # This is a simple replacement and won't affect hyphens within words.
        return text.replace(" - ", "\n- ")

    async def translate_batch(
        self,
        batch: List[SubtitleBlock],
        batch_num: int,
        total_batches: int,
        target_language: str,
    ) -> List[TranslationResult]:
        async with self.semaphore:
            texts = [block.text for block in batch]
            translated_texts = await self.tran_api.translate(texts, target_language)

            results = [
                TranslationResult(
                    number=block.number,
                    timestamp=block.timestamp,
                    text=self.post_process_translation(translated_text),
                    original_text=block.text,
                )
                for block, translated_text in zip(batch, translated_texts)
            ]

            logger.info(f"Batch {batch_num}/{total_batches}: Completed")
            return results

    async def translate_file(self, input_file: str, output_file: str, batch_size: int, target_language: str, model_name: str, no_header: bool):
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
            self.translate_batch(batch, i + 1, len(batches), target_language)
            for i, batch in enumerate(batches)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                raise RuntimeError(f"Batch {i+1} failed: {str(result)}")
            else:
                translated_blocks.extend(result)

        # Sort results by subtitle number
        translated_blocks.sort(key=lambda x: x.number)

        # Add header block if not disabled
        if translated_blocks and not no_header:
            header_text = f"Translated by srtgpt (model: {model_name})"
            first_block_start_ms = parse_timestamp(
                translated_blocks[0].timestamp.split(" --> ")[0]
            )

            header_end_ms = 5000  # Default 5 seconds
            if 0 < first_block_start_ms < header_end_ms:
                header_end_ms = first_block_start_ms - 1

            header_timestamp = f"00:00:00,000 --> {ms_to_timestamp(header_end_ms)}"

            header_block = TranslationResult(
                number=0,  # Placeholder, will be renumbered
                timestamp=header_timestamp,
                text=header_text,
                original_text="",
            )

            translated_blocks.insert(0, header_block)

            # Renumber all blocks starting from 1
            final_blocks = [
                block._replace(number=i + 1) for i, block in enumerate(translated_blocks)
            ]
        else:
            final_blocks = translated_blocks

        # Write final result to file
        self.write_file(output_file, final_blocks)

        logger.info(
            f"Translation completed: {len(final_blocks)} subtitles"
        )


def parse_srt_filename(input_path: str) -> Tuple[Path, str, str]:
    """Parse SRT filename to extract folder, name, and language."""
    path = Path(input_path)
    
    if path.suffix != ".srt":
        raise ValueError(f"Not .srt file: {input_path}")
    
    # Remove .srt extension
    stem = path.stem
    
    # Remove .hi if present (hearing impaired)
    stem = stem.replace(".hi", "")
    
    # Extract language (last part after .)
    parts = stem.rsplit(".", 1)
    if len(parts) == 2:
        name, lang = parts
    else:
        name = stem
        lang = ""
    
    return path.parent, name, lang


def parse_args():
    parser = argparse.ArgumentParser(
        description="Translate SRT subtitles using an OpenAI-compatible API."
    )
    parser.add_argument("input_path", help="Input SRT file path or folder containing SRT files")
    parser.add_argument(
        "--api-base",
        required=True,
        help="OpenAI API proxy server base URL",
    )
    parser.add_argument(
        "--api-key",
        required=True,
        help="API key for authentication",
    )
    parser.add_argument(
        "--target-language",
        default="Traditional Chinese",
        help="The target language for translation (e.g., 'English', 'Japanese', 'Traditional Chinese')",
    )
    parser.add_argument(
        "--target-lang-code",
        default="zh-TW",
        help="The target language code for the output filename (e.g., 'en', 'ja', 'zh-TW')",
    )
    parser.add_argument(
        "--also-skip",
        nargs="*",
        default=[],
        help="Also skip translating if files with these language codes exist (e.g., for target 'zh-TW', you might want to skip if 'zh' exists).",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="LLM model",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
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
        "--no-header",
        action="store_true",
        help="Do not add a header with translation info to the SRT file",
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
        target_lang_code = args.target_lang_code
        skip_codes = [target_lang_code] + args.also_skip

        # Skip if input file is already in one of the target languages
        if lang in skip_codes:
            logger.info(f"Input subtitle language '{lang}' is in the list of languages to skip. Skipping.")
            return

        if not args.force:
            # check existing translated subtitles
            for code in skip_codes:
                if (folder / f"{name}.{code}.srt").exists():
                    logger.info(
                        f"Found existing translated file '{name}.{code}.srt'. Skipping. Use --force to translate anyway."
                    )
                    return

        try:
            await tran_srt.translate_file(
                input_file, str(folder / f"{name}.{target_lang_code}.srt"), args.batch_size, args.target_language, args.model, args.no_header
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
    
    # Validate input path
    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        return 1
    
    # Initialize translation API and SRT handler with context manager
    async with TranAPI(
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        max_retries=args.max_retries,
        timeout=args.timeout,
        initial_delay=args.initial_delay,
    ) as tran_api:
        tran_srt = TranSRT(
            tran_api=tran_api,
            max_concurrent_requests=args.max_concurrent,
            filter_bad_words=args.filter_bad_words,
        )

        try:
            if input_path.is_file():
                # Process single file
                if not input_path.name.endswith(".srt"):
                    logger.error(f"Not .srt file: {input_path}")
                    return 1
                success = await process_file(str(input_path), tran_srt, args)
                return 0 if success else 1
                
            elif input_path.is_dir():
                # Process all .srt files in directory
                srt_files = list(input_path.glob("**/*.srt"))
                if not srt_files:
                    logger.error(f"No .srt files found in {input_path}")
                    return 1
                
                logger.info(f"Found {len(srt_files)} .srt files to process")
                failed_files = []
                
                for srt_file in srt_files:
                    logger.info(f"Processing {srt_file}")
                    success = await process_file(str(srt_file), tran_srt, args)
                    if success is False:
                        failed_files.append(srt_file)
                
                if failed_files:
                    logger.error(f"Failed to process {len(failed_files)} files:")
                    for f in failed_files:
                        logger.error(f"  - {f}")
                    return 1
                
                logger.info("All files processed successfully")
                return 0
            else:
                logger.error(f"Invalid input path: {input_path}")
                return 1
                
        except KeyboardInterrupt:
            logger.warning("Process interrupted by user")
            return 130  # Standard exit code for SIGINT
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            return 1


if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code or 0)
