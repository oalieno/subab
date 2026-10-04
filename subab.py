import argparse
import asyncio
import json
import logging
import random
import re
import secrets
import sys
from datetime import timedelta
from pathlib import Path

import httpx
import json_repair
import srt


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


class RateLimitError(Exception):
    """Rate limit exceeded, should wait before retrying."""

    def __init__(self, message: str, retry_after: float = None):
        super().__init__(message)
        self.retry_after = retry_after


class ServerError(Exception):
    """Server error, should retry."""

    ...


class LLMAPI:
    def __init__(
        self,
        api_base: str,
        api_key: str,
        model: str,
        max_retries: int = 3,
        timeout: float = 60.0,
        initial_delay: float = 30.0,
        max_concurrent: int = 5,
        provider_order: list[str] | None = None,
        no_reasoning: bool = False,
    ):
        self.api_endpoint = f"{api_base}/v1/chat/completions"
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.timeout = timeout
        self.provider_order = provider_order
        self.no_reasoning = no_reasoning
        self._client = None
        self._semaphore = asyncio.Semaphore(max_concurrent)

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

    async def _backoff(
        self, attempt: int, reason: str, *, retry_after: float | None = None
    ):
        base_wait = (
            retry_after
            if retry_after is not None
            else self.initial_delay * (2**attempt)
        )
        # Add small jitter to avoid thundering herd
        jitter = 0.8 + (random.random() * 0.4)
        wait_time = base_wait * jitter
        logger.warning(
            f"{reason}. Retrying in {wait_time:.1f}s (base {base_wait:.1f}s, attempt {attempt + 1}/{self.max_retries})"
        )
        await asyncio.sleep(wait_time)

    async def call(self, content: str) -> str:
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                payload = {
                    "model": self.model,
                    "stream": False,
                    "temperature": 0.2,
                    "top_p": 1,
                    "messages": [{"role": "user", "content": content}],
                }
                if self.provider_order:
                    # OpenRouter provider routing: pin to specific providers
                    # and never fall back to a random one.
                    payload["provider"] = {
                        "order": self.provider_order,
                        "allow_fallbacks": False,
                    }
                if self.no_reasoning:
                    # Hybrid reasoning models (e.g. deepseek-v4) think by
                    # default, which is 5-10x slower and unnecessary for
                    # subtitle translation.
                    payload["reasoning"] = {"enabled": False}

                # Strictly limit concurrent API requests
                await self._semaphore.acquire()
                try:
                    # Hard deadline on the whole request. OpenRouter keeps
                    # non-streaming connections alive with periodic whitespace,
                    # which resets httpx's read timeout, so a stuck provider
                    # can hang forever without this.
                    response = await asyncio.wait_for(
                        self.client.post(
                            self.api_endpoint,
                            headers={
                                "Content-Type": "application/json",
                                "Authorization": f"Bearer {self.api_key}",
                            },
                            json=payload,
                        ),
                        timeout=self.timeout * 3,
                    )
                finally:
                    self._semaphore.release()

                # Handle different HTTP status codes
                if response.status_code == 429:
                    # Rate limit exceeded
                    retry_after = None
                    try:
                        retry_after = float(response.headers.get("Retry-After"))
                    except (TypeError, ValueError):
                        pass

                    raise RateLimitError(
                        "Rate limit exceeded (429)",
                        retry_after,
                    )

                elif 400 <= response.status_code < 500:
                    # Client errors (except 429) - don't retry
                    error_msg = f"Client error {response.status_code}: {response.text}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)

                elif 500 <= response.status_code < 600:
                    # Server errors - retry with backoff
                    raise ServerError(f"Server error {response.status_code}")

                # Successful response
                response.raise_for_status()

                # Guard: empty/whitespace-only response probably means overload context
                # Treat as empty response, let our adaptive batching handle it
                if not response.text or response.text.strip() == "":
                    return ""

                # Parse response body as JSON
                try:
                    result = response.json()
                except json.JSONDecodeError as e:
                    body = response.text or ""
                    logger.error(
                        "Invalid JSON response (status=%s, len=%s, head=%r)",
                        response.status_code,
                        len(body),
                        body[:200],
                    )
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

            except (httpx.TimeoutException, asyncio.TimeoutError) as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    await self._backoff(attempt, "Request timeout")

            except httpx.NetworkError as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    await self._backoff(attempt, f"Network error: {str(e)}")

            except RateLimitError as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    await self._backoff(attempt, str(e), retry_after=e.retry_after)

            except ServerError as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    await self._backoff(attempt, str(e))

            except RuntimeError as e:
                # Don't retry RuntimeError (client errors, validation errors)
                raise e

            except Exception as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    await self._backoff(attempt, f"Unexpected error: {str(e)}")

        raise RuntimeError(
            f"API failed after {self.max_retries} attempts. Last error: {last_exception}"
        )


class SubtitleTranslator:
    def __init__(
        self,
        llm_api: LLMAPI,
        tag_mode: str = "opaque",
        filter_bad_words: bool = False,
        karaoke_policy: str = "remove",
    ):
        self.llm_api = llm_api
        self.tag_mode = tag_mode if tag_mode in {"opaque", "numeric"} else "opaque"
        self.filter_bad_words = filter_bad_words
        self.karaoke_policy = (
            karaoke_policy
            if karaoke_policy in {"skip", "remove", "translate"}
            else "remove"
        )

    @staticmethod
    def normalize_text(text: str) -> str:
        return re.sub(r"\s+", "", text).strip()

    @staticmethod
    def filter_bad(text: str) -> str:
        bad_words = {
            "dick": "dic*",
            "pussy": "pu**y",
            "blow job": "blo* job",
            "rape": "rap*",
            "orgasm": "orgas*",
            "had sex": "haɗ seҳ",
            "have sex": "hav* seҳ",
            "masturbate": "masturbat*",
        }

        for word, censored in bad_words.items():
            # Case-insensitive replacement
            text = re.sub(re.escape(word), censored, text, flags=re.IGNORECASE)
        return text

    def preprocess_subtitles(self, subtitles: list[srt.Subtitle]) -> list[srt.Subtitle]:
        seen: set[tuple] = set()
        result: list[srt.Subtitle] = []

        for sub in subtitles:
            # Normalize content
            sub.content = self.normalize_text(sub.content)

            # Skip empty subtitles
            if sub.content == "":
                continue

            # Deduplicate exact subtitles
            key = (sub.start, sub.end, sub.content)
            if key in seen:
                continue
            seen.add(key)

            # Filter bad words
            if self.filter_bad_words:
                sub.content = self.filter_bad(sub.content)

            result.append(sub)

        return result

    @staticmethod
    def postprocess_subtitles(subtitles: list[srt.Subtitle]) -> list[srt.Subtitle]:
        for sub in subtitles:
            # Split dialogues like "- Hello - World" into separate lines for readability.
            # This is a simple replacement and won't affect hyphens within words.
            sub.content = sub.content.replace(" - ", "\n- ")
        return subtitles

    @staticmethod
    def duration_seconds(sub: srt.Subtitle) -> float:
        return max(0.0, (sub.end - sub.start).total_seconds())

    def is_potential_karaoke(self, sub: srt.Subtitle) -> bool:
        return len(sub.content) <= 4 or self.duration_seconds(sub) <= 0.6

    def compute_karaoke_mask(self, subtitles: list[srt.Subtitle]) -> list[bool]:
        # Phase 1: mark potential lines by simple per-line heuristic
        potential = [self.is_potential_karaoke(s) for s in subtitles]

        # Phase 2: confirm by local density within a time window
        window_s = 15.0
        count_threshold = 10
        n = len(subtitles)
        mask: list[bool] = [False] * n
        starts = [s.start.total_seconds() for s in subtitles]
        for i in range(n):
            # skip if not potential karaoke
            if not potential[i]:
                continue

            # expand window indices based on time distance
            t0, l, r = starts[i], i, i
            while l - 1 >= 0 and t0 - starts[l - 1] <= window_s:
                l -= 1
            while r + 1 < n and starts[r + 1] - t0 <= window_s:
                r += 1

            # count potential karaoke in the window
            count = 0
            for j in range(l, r + 1):
                if potential[j]:
                    count += 1
            # confirm by local density
            mask[i] = count >= count_threshold

        return mask

    def build_prompt(
        self,
        target_language: str,
        tagged_inputs: list[str],
        glossary_entries: list[tuple[str, str]] | None = None,
    ) -> str:
        if self.tag_mode == "numeric":
            examples = (
                '- Input: ["0000|A line broken into", "0001|two parts."]\n'
                '  Output: ["0000|<translation of first>", "0001|<translation of second>"]\n\n'
                '- Input: ["0000|- A single line dialogue. - With two speakers."]\n'
                '  Output: ["0000|<translation of the whole line>"]'
            )
            rules_prefix = (
                "2.  Prefix Preservation: Each input string starts with a numeric prefix like `0000|`. "
                "Copy the SAME prefix at the start of the corresponding output string."
            )
        else:
            examples = (
                '- Input: ["aa11|A line broken into", "cc33|two parts."]\n'
                '  Output: ["aa11|<translation of first>", "cc33|<translation of second>"]\n\n'
                '- Input: ["aa11|- A single line dialogue. - With two speakers."]\n'
                '  Output: ["aa11|<translation of the whole line>"]'
            )
            rules_prefix = (
                "2.  Prefix Preservation: Each input string starts with an opaque token like `aa11|`. "
                "Copy the SAME token at the start of the corresponding output string (exact characters)."
            )

        glossary = ""
        if glossary_entries:
            glossary_lines = "\n".join(
                f'- "{source}" -> "{target}"'
                for source, target in glossary_entries
            )
            glossary = f"""
### Glossary:
Use these preferred translations when the source term appears. Match context naturally and do not force a glossary entry when the source term clearly has a different meaning.
{glossary_lines}
"""

        template = f"""Translate the subtitles into {{target_language}}, preserving the original line structure.

### Rules:
1.  1-to-1 Mapping: Output MUST be a JSON array with the SAME number of strings. Do not merge or split.
{rules_prefix}
3.  Translation Quality: Match tone and meaning. Adapt idioms. Translate names.
4.  Strict JSON Output: Output ONLY a valid JSON array of strings.
{{glossary}}

### Examples:
{examples}

### JSON Output Format:
```json
{{{{ "type": "array", "items": {{{{ "type": "string" }}}} }}}}
```

### Your Task:
Translate the following input into {{target_language}}:
```json
{{tagged_inputs}}
```"""

        return template.format(
            target_language=target_language,
            tagged_inputs=json.dumps(tagged_inputs, ensure_ascii=False),
            glossary=glossary,
        )

    def _build_header(self, first_start: timedelta, model_name: str) -> srt.Subtitle:
        return srt.Subtitle(
            index=0,
            start=timedelta(seconds=0),
            end=min(
                timedelta(seconds=5),
                max(first_start - timedelta(milliseconds=1), timedelta(seconds=0)),
            ),  # not overlapping with the first subtitle, no less than 0 seconds, no more than 5 seconds
            content=f"Translated by SubAB (model: {model_name})",
        )

    def make_ids(self, n: int) -> list[str]:
        # Numeric IDs
        if self.tag_mode == "numeric":
            return [f"{i:04d}" for i in range(n)]
        # Opaque tokens (default): ensure uniqueness
        ids: list[str] = []
        seen: set[str] = set()
        while len(ids) < n:
            token = secrets.token_hex(2)
            if token in seen:
                continue
            seen.add(token)
            ids.append(token)
        return ids

    async def translate(
        self,
        texts: list[str],
        target_language: str,
        glossary_entries: list[tuple[str, str]] | None = None,
    ) -> list[str]:
        try:
            ids = self.make_ids(len(texts))
            tagged_inputs = [f"{id}|{text}" for id, text in zip(ids, texts)]

            prompt = self.build_prompt(target_language, tagged_inputs, glossary_entries)
            response = await self.llm_api.call(prompt)

            tagged_outputs = json_repair.loads(response)

            # Check JSON schema
            if (
                not isinstance(tagged_outputs, list)
                or not tagged_outputs
                or not all(isinstance(text, str) for text in tagged_outputs)
            ):
                raise TranslationError(
                    f"Not a valid array of strings: {tagged_outputs}"
                )

            outputs = [text.partition("|")[2] for text in tagged_outputs]

            input_ids = {i.partition("|")[0] for i in tagged_inputs}
            output_ids = {o.partition("|")[0] for o in tagged_outputs}

            # Attempt repair when exactly one line is missing (likely merge)
            # This is the most common case for merge errors
            if (
                len(tagged_inputs) == len(tagged_outputs) + 1
                and len(input_ids - output_ids) == 1
                and len(tagged_inputs) >= 10
            ):
                missing_id = list(input_ids - output_ids)[0]
                index = ids.index(missing_id)
                start = max(0, index - 2)
                end = min(index + 3, len(texts))

                logger.info(f"Repairing missing line at index {index}")

                return (
                    outputs[:start]
                    + await self.translate(
                        texts[start:end],
                        target_language,
                        glossary_entries,
                    )
                    + outputs[end:]
                )

            # Check if the number of outputs is the same as the number of inputs
            if len(tagged_outputs) != len(tagged_inputs):
                raise TranslationError(
                    f"Translation count mismatch: got {len(tagged_outputs)}, expected {len(tagged_inputs)}"
                )

            # Check if the IDs are exactly the same in order
            for i, o in zip(tagged_inputs, tagged_outputs):
                if i.partition("|")[0] != o.partition("|")[0]:
                    raise TranslationError(f"ID mismatch: {i} -> {o}")

            return outputs

        except TranslationError as e:
            logger.info(f"Error in translation: {str(e)}. Retrying...")

        if len(texts) == 1:
            raise RuntimeError("Translation failed after many tries...")

        # Adaptive batching: split the input into two halves and translate them separately
        # This will fix the problem of too many lines being translated at once
        return await self.translate(
            texts[: len(texts) // 2], target_language, glossary_entries
        ) + await self.translate(
            texts[len(texts) // 2 :], target_language, glossary_entries
        )

    async def translate_batch(
        self,
        batch: list[srt.Subtitle],
        batch_num: int,
        total_batches: int,
        target_language: str,
        glossary_entries: list[tuple[str, str]] | None,
    ) -> list[srt.Subtitle]:
        translated_texts = await self.translate(
            [sub.content for sub in batch], target_language, glossary_entries
        )

        for sub, translated_text in zip(batch, translated_texts):
            sub.content = translated_text

        logger.info(f"Batch {batch_num}/{total_batches}: Completed")

        return batch

    async def translate_file(
        self,
        input_file: str,
        output_file: str,
        batch_size: int,
        target_language: str,
        model_name: str,
        no_header: bool,
        glossary_entries: list[tuple[str, str]] | None = None,
        parsed_subtitles: list[srt.Subtitle] | None = None,
    ):
        if parsed_subtitles is None:
            with open(input_file, encoding="utf-8", errors="replace") as file:
                subtitles = list(srt.parse(file.read()))
        else:
            subtitles = parsed_subtitles

        # preprocess subtitles
        subtitles = self.preprocess_subtitles(subtitles)

        # Check if the input file is empty
        if not subtitles:
            raise ValueError(f"Input file {input_file} is empty")

        # karaoke handling (pre-batch): detect and apply policy
        if self.karaoke_policy in {"skip", "remove"}:
            karaoke_mask = self.compute_karaoke_mask(subtitles)
            if all(karaoke_mask):
                raise ValueError(
                    "SRT file appears to be all short lines, possibly broken."
                )
        else:
            karaoke_mask = [False] * len(subtitles)

        # policy: remove
        if self.karaoke_policy == "remove":
            subtitles = [s for i, s in enumerate(subtitles) if not karaoke_mask[i]]
            pending_subtitles = subtitles
        # policy: skip
        elif self.karaoke_policy == "skip":
            pending_subtitles = [
                s for i, s in enumerate(subtitles) if not karaoke_mask[i]
            ]
        # policy: translate
        else:
            pending_subtitles = list(subtitles)

        # Prepare batches (balanced) from pending_subtitles
        batches = []
        n_pending = len(pending_subtitles)
        num_groups = max(1, (n_pending + batch_size - 1) // batch_size)  # ceil
        base = n_pending // num_groups
        extra = n_pending % num_groups
        start = 0
        for i in range(num_groups):
            size = base + (1 if i < extra else 0)
            batches.append(pending_subtitles[start : start + size])
            start += size

        # Process all batches, replace srt in place
        translated_batches = await asyncio.gather(
            *[
                self.translate_batch(
                    batch, i + 1, len(batches), target_language, glossary_entries
                )
                for i, batch in enumerate(batches)
            ],
            return_exceptions=True,
        )

        # Check for errors in any batch
        for i, batch in enumerate(translated_batches):
            if isinstance(batch, Exception):
                raise RuntimeError(f"Batch {i + 1} failed: {batch}")

        # postprocess subtitles
        subtitles = self.postprocess_subtitles(subtitles)

        # Add header subtitle if not disabled
        if not no_header and subtitles:
            subtitles.insert(0, self._build_header(subtitles[0].start, model_name))

        # Renumber all subtitles starting from 1
        for i, sub in enumerate(subtitles):
            sub.index = i + 1

        # Write subtitles to file
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(srt.compose(subtitles))

        logger.info(f"Translation completed: {len(subtitles)} subtitles")


VIDEO_EXTENSIONS = (".mkv", ".mp4", ".avi", ".m4v", ".ts", ".webm", ".mov", ".wmv")
SUBTITLE_FLAGS = {"hi", "sdh", "cc", "forced"}
LANG_CODE_RE = re.compile(r"^[A-Za-z]{2,3}([-_][A-Za-z0-9]{2,4})?$")


def parse_srt_filename(input_path: str) -> tuple[Path, str, str]:
    """Parse SRT filename to extract folder, name, and language."""
    path = Path(input_path)

    if path.suffix != ".srt":
        raise ValueError(f"Not .srt file: {input_path}")

    # Remove .srt extension
    stem = path.stem

    # Release names contain dots (e.g. "[AAC 2.0]"), so prefer the video file
    # next to the subtitle to tell where the name ends and the tags begin.
    try:
        video_stems = [
            p.stem
            for p in path.parent.iterdir()
            if p.suffix.lower() in VIDEO_EXTENSIONS
        ]
    except OSError:
        video_stems = []
    matches = [v for v in video_stems if stem == v or stem.startswith(v + ".")]
    if matches:
        name = max(matches, key=len)
        tags = [
            t
            for t in stem[len(name) :].split(".")
            if t and t.lower() not in SUBTITLE_FLAGS
        ]
        return path.parent, name, tags[-1] if tags else ""

    # No matching video: drop trailing flags (.hi, .forced, ...) and treat the
    # last part as the language only if it looks like a language code.
    parts = stem.split(".")
    while len(parts) > 1 and parts[-1].lower() in SUBTITLE_FLAGS:
        parts.pop()
    if len(parts) > 1 and LANG_CODE_RE.match(parts[-1]):
        return path.parent, ".".join(parts[:-1]), parts[-1]
    return path.parent, ".".join(parts), ""


GLOSSARY_EXTENSIONS = (".json", ".txt", ".tsv", ".csv", ".md")
MAX_GLOSSARY_ENTRIES = 400
MAX_AUTO_GLOSSARY_ENTRIES = 80


def dedupe_glossary_entries(
    entries: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    deduped: dict[str, str] = {}
    order: list[str] = []

    for source, target in entries:
        source = source.strip()
        target = target.strip()
        if not source or not target:
            continue
        if source not in deduped:
            order.append(source)
        # Later files override earlier files, so series-specific entries can
        # override global entries.
        deduped[source] = target

    return [(source, deduped[source]) for source in order][:MAX_GLOSSARY_ENTRIES]


def entries_from_mapping(mapping: dict) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for source, target in mapping.items():
        if isinstance(target, str):
            entries.append((str(source), target))
    return entries


def parse_json_glossary(path: Path) -> list[tuple[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))

    if isinstance(data, dict):
        for key in ("terms", "glossary", "entries"):
            if key in data:
                data = data[key]
                break

    if isinstance(data, dict):
        return entries_from_mapping(data)

    if isinstance(data, list):
        entries: list[tuple[str, str]] = []
        for item in data:
            if isinstance(item, dict):
                source = item.get("source") or item.get("from") or item.get("term")
                target = item.get("target") or item.get("to") or item.get("translation")
                if isinstance(source, str) and isinstance(target, str):
                    entries.append((source, target))
            elif (
                isinstance(item, list)
                and len(item) >= 2
                and isinstance(item[0], str)
                and isinstance(item[1], str)
            ):
                entries.append((item[0], item[1]))
        return entries

    return []


def parse_text_glossary(path: Path) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        source = target = ""
        for separator in ("\t", "=>", "->", "=", ":"):
            if separator in line:
                source, target = line.split(separator, 1)
                break

        if source and target:
            entries.append((source.strip(), target.strip()))

    return entries


def parse_glossary_file(path: Path) -> list[tuple[str, str]]:
    if path.suffix == ".json":
        return parse_json_glossary(path)
    return parse_text_glossary(path)


def matching_glossary_files(input_file: str, glossary_dir: str | None) -> list[Path]:
    if not glossary_dir:
        return []

    root = Path(glossary_dir)
    if not root.is_dir():
        logger.warning(f"Glossary directory does not exist: {root}")
        return []

    input_path = Path(input_file)
    stems: list[str] = ["global"]
    for parent in reversed(input_path.parent.parents):
        stems.append(parent.name)
    stems.append(input_path.parent.name)

    files: list[Path] = []
    seen: set[Path] = set()
    for stem in stems:
        if not stem:
            continue
        manual_paths: list[Path] = []
        auto_paths: list[Path] = []
        for extension in GLOSSARY_EXTENSIONS:
            manual_path = root / f"{stem}{extension}"
            auto_path = root / f"{stem}.auto{extension}"
            if manual_path.is_file():
                manual_paths.append(manual_path)
            if auto_path.is_file():
                auto_paths.append(auto_path)

        # A hand-written glossary for the same stem replaces its auto-generated
        # draft entirely. This keeps one-off auto terms from leaking forward.
        for path in manual_paths or auto_paths:
            if path not in seen:
                files.append(path)
                seen.add(path)

    return files


def series_glossary_files(input_file: str, glossary_dir: str | None) -> list[Path]:
    if not glossary_dir:
        return []

    root = Path(glossary_dir)
    if not root.is_dir():
        return []

    series_name = Path(input_file).parent.name
    files: list[Path] = []
    for extension in GLOSSARY_EXTENSIONS:
        for suffix in ("", ".auto"):
            path = root / f"{series_name}{suffix}{extension}"
            if path.is_file():
                files.append(path)
    return files


def load_glossary_entries(
    input_file: str, glossary_dir: str | None
) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    files = matching_glossary_files(input_file, glossary_dir)

    for path in files:
        try:
            entries.extend(parse_glossary_file(path))
        except Exception as e:
            logger.warning(f"Failed to parse glossary file {path}: {e}")

    entries = dedupe_glossary_entries(entries)
    if entries:
        logger.info(
            "Loaded %s glossary entries from %s",
            len(entries),
            ", ".join(str(path) for path in files),
        )
    return entries


def sample_subtitle_texts(
    subtitles: list[srt.Subtitle], max_lines: int
) -> list[str]:
    if max_lines <= 0 or len(subtitles) <= max_lines:
        return [sub.content for sub in subtitles]

    # Keep coverage across the whole episode/movie instead of only sampling the
    # beginning, where OP/signs can dominate.
    indexes = sorted(
        {
            round(i * (len(subtitles) - 1) / (max_lines - 1))
            for i in range(max_lines)
        }
    )
    return [subtitles[i].content for i in indexes]


def build_auto_glossary_prompt(
    target_language: str,
    texts: list[str],
    max_entries: int,
    *,
    series_title: str,
    media_title: str,
) -> str:
    return f"""Extract a translation glossary for subtitle translation into {target_language}.

Return ONLY a valid JSON object mapping source terms to preferred {target_language} translations.

Media context:
- Series/movie folder: {series_title}
- Subtitle/media file: {media_title}

Rules:
1. Include only named entities and series-specific terms: character names, family names, organizations, places, spells, formal titles, artifacts, races, and recurring fictional concepts.
2. Do not include common dictionary words, ordinary verbs/adjectives, generic phrases, speaker labels, or full sentences.
3. Prefer established/natural {target_language} translations when obvious.
4. Keep source terms exactly as they appear in the subtitles.
5. Limit to at most {max_entries} entries.
6. Exclude complete dialogue lines, commands, greetings, interjections, and one-off descriptions.
7. The examples below show how to extract terms from subtitle samples. Never copy an example term unless it appears in the media context or subtitle sample.

Extraction examples:
```json
[
  {{
    "subtitle_sample": [
      "My name is Natsuki Subaru!",
      "Crusch-sama, the White Whale is approaching.",
      "Rem and Ram are waiting in Lugunica.",
      "Thank you.",
      "Run!"
    ],
    "expected_glossary": {{
      "Natsuki Subaru": "菜月昴",
      "Crusch-sama": "庫珥修大人",
      "White Whale": "白鯨",
      "Rem": "雷姆",
      "Ram": "拉姆",
      "Lugunica": "露格尼卡"
    }}
  }},
  {{
    "subtitle_sample": [
      "Voice",
      "Girl",
      "Knight A",
      "I know.",
      "Things are getting better.",
      "What happened?",
      "Get in trouble.",
      "Close your eyes."
    ],
    "expected_glossary": {{}}
  }}
]
```

Subtitles:
```json
{json.dumps(texts, ensure_ascii=False)}
```"""


def looks_like_dialogue_line(source: str) -> bool:
    source = source.strip()
    if not source:
        return True
    if re.search(r"[.!?…。！？]$", source):
        return True
    if "..." in source or "…" in source:
        return True

    words = source.split()
    if len(words) > 5:
        return True

    lower = source.lower()
    if re.search(r"\b(i|you|we|they|he|she|it|my|your|our|me|us)\b", lower):
        return True

    return False


def looks_like_named_or_fictional_term(source: str) -> bool:
    source = source.strip()
    if not source:
        return False

    if re.search(r"[-']", source):
        return True
    if re.search(r"\b[A-Z]{2,}\b", source):
        return True
    if re.search(r"\b[A-Z][a-z]+(?:\s+(?:of|the|van|von|de|da|du|[A-Z][a-z]+))*\b", source):
        return True

    return False


def filter_auto_glossary_entries(
    entries: list[tuple[str, str]],
    max_entries: int,
    *,
    source_context: str | None = None,
) -> list[tuple[str, str]]:
    filtered: list[tuple[str, str]] = []
    normalized_context = source_context.lower() if source_context else ""
    generic_speaker_labels = {
        "boy",
        "girl",
        "man",
        "woman",
        "old man",
        "young man",
        "young woman",
        "child",
        "children",
        "merchant",
        "vendor",
        "guard",
        "soldier",
        "knight",
        "voice",
    }

    for source, target in dedupe_glossary_entries(entries):
        if looks_like_dialogue_line(source):
            continue
        if not looks_like_named_or_fictional_term(source):
            continue
        if normalized_context and source.lower() not in normalized_context:
            continue
        if source.strip().lower() in generic_speaker_labels:
            continue
        if re.fullmatch(r"[A-Z][a-z]+ [A-Z]", source.strip()):
            continue
        filtered.append((source, target))
        if len(filtered) >= max_entries:
            break
    return filtered


def write_text_glossary(
    path: Path,
    entries: list[tuple[str, str]],
    *,
    input_file: str,
    model_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Auto-generated by SubAB (model: {model_name})",
        f"# Source: {input_file}",
        "# Review recommended.",
        "",
    ]
    lines.extend(f"{source} => {target}" for source, target in entries)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


async def generate_auto_glossary(
    input_file: str,
    subtitles: list[srt.Subtitle],
    llm_api: LLMAPI,
    target_language: str,
    model_name: str,
    glossary_dir: str,
    max_lines: int,
    max_entries: int,
) -> list[tuple[str, str]]:
    root = Path(glossary_dir)
    series_name = Path(input_file).parent.name
    output_path = root / f"{series_name}.auto.txt"

    texts = sample_subtitle_texts(subtitles, max_lines)
    if not texts:
        return []

    input_path = Path(input_file)
    prompt = build_auto_glossary_prompt(
        target_language,
        texts,
        max_entries,
        series_title=input_path.parent.name,
        media_title=input_path.name,
    )
    response = await llm_api.call(prompt)
    data = json_repair.loads(response)

    if isinstance(data, dict):
        entries = entries_from_mapping(data)
    elif isinstance(data, list):
        entries = []
        for item in data:
            if isinstance(item, dict):
                source = item.get("source") or item.get("from") or item.get("term")
                target = item.get("target") or item.get("to") or item.get("translation")
                if isinstance(source, str) and isinstance(target, str):
                    entries.append((source, target))
            elif (
                isinstance(item, list)
                and len(item) >= 2
                and isinstance(item[0], str)
                and isinstance(item[1], str)
            ):
                entries.append((item[0], item[1]))
    else:
        raise RuntimeError(f"Unexpected auto glossary response: {data}")

    source_context = "\n".join(
        [
            input_path.parent.name,
            input_path.name,
            *(sub.content for sub in subtitles),
        ]
    )
    entries = filter_auto_glossary_entries(
        entries,
        max_entries,
        source_context=source_context,
    )
    if entries:
        write_text_glossary(
            output_path,
            entries,
            input_file=input_file,
            model_name=model_name,
        )
        logger.info(
            "Generated auto glossary with %s entries: %s",
            len(entries),
            output_path,
        )
    else:
        logger.info("Auto glossary generation returned no entries")

    return entries


def parse_args():
    parser = argparse.ArgumentParser(
        description="Translate SRT subtitles using an OpenAI-compatible API."
    )
    parser.add_argument(
        "input_path", help="Input SRT file path or folder containing SRT files"
    )
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
        "--provider",
        default=None,
        help=(
            "Comma-separated OpenRouter provider slugs to pin, in order of "
            "preference (e.g. 'relace/fp4,novita/fp8'). Disables fallback to "
            "other providers."
        ),
    )
    parser.add_argument(
        "--no-reasoning",
        action="store_true",
        help=(
            "Disable thinking/reasoning on hybrid reasoning models "
            "(OpenRouter 'reasoning: {enabled: false}'). Much faster for "
            "translation."
        ),
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=400,
        help="Maximum number of subtitles to translate in one batch",
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
        default=30.0,
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
        "--tag-mode",
        choices=["opaque", "numeric"],
        default="opaque",
        help="Token tagging scheme to enforce 1:1 mapping (opaque|numeric)",
    )
    parser.add_argument(
        "--glossary-dir",
        default=None,
        help=(
            "Optional directory containing glossary files. Loads global.* plus "
            "files named after path components, such as the series folder name."
        ),
    )
    parser.add_argument(
        "--auto-glossary",
        action="store_true",
        help=(
            "Generate a series .auto.txt glossary when no series glossary exists. "
            "Requires --glossary-dir and adds one extra API call for the first file in a series."
        ),
    )
    parser.add_argument(
        "--auto-glossary-max-lines",
        type=int,
        default=160,
        help="Maximum number of sampled subtitle lines used to generate an auto glossary.",
    )
    parser.add_argument(
        "--auto-glossary-max-entries",
        type=int,
        default=MAX_AUTO_GLOSSARY_ENTRIES,
        help="Maximum number of entries generated for an auto glossary.",
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
        "--karaoke-policy",
        choices=["skip", "remove", "translate"],
        default="remove",
        help="How to handle karaoke-like short lines (skip=keep as-is, remove=drop, translate=translate)",
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


async def process_file(input_file: str, subtitle_translator: SubtitleTranslator, args):
    try:
        folder, name, lang = parse_srt_filename(input_file)
        target_lang_code = args.target_lang_code
        skip_codes = [target_lang_code] + args.also_skip

        # Skip if input file is already in one of the target languages
        if lang in skip_codes:
            logger.info(
                f"Input subtitle language '{lang}' is in the list of languages to skip. Skipping."
            )
            return

        if not args.force:
            # check existing translated subtitles
            for code in skip_codes:
                for extra in ["", ".hi"]:
                    if (folder / f"{name}.{code}{extra}.srt").exists():
                        logger.info(
                            f"Found existing translated file '{name}.{code}{extra}.srt'. Skipping. Use --force to translate anyway."
                        )
                        return

        logger.info(f"Processing {input_file}")
        glossary_entries = load_glossary_entries(input_file, args.glossary_dir)

        with open(input_file, encoding="utf-8", errors="replace") as file:
            parsed_subtitles = list(srt.parse(file.read()))

        if (
            args.auto_glossary
            and args.glossary_dir
            and not series_glossary_files(input_file, args.glossary_dir)
        ):
            try:
                auto_entries = await generate_auto_glossary(
                    input_file=input_file,
                    subtitles=parsed_subtitles,
                    llm_api=subtitle_translator.llm_api,
                    target_language=args.target_language,
                    model_name=args.model,
                    glossary_dir=args.glossary_dir,
                    max_lines=args.auto_glossary_max_lines,
                    max_entries=args.auto_glossary_max_entries,
                )
                glossary_entries = dedupe_glossary_entries(
                    glossary_entries + auto_entries
                )
            except Exception as e:
                logger.warning(
                    "Auto glossary generation failed; continuing without a generated glossary: %s",
                    e,
                )
        elif args.auto_glossary and not args.glossary_dir:
            logger.warning("--auto-glossary ignored because --glossary-dir is not set")

        await subtitle_translator.translate_file(
            input_file,
            str(folder / f"{name}.{target_lang_code}.srt"),
            args.max_batch_size,
            args.target_language,
            args.model,
            args.no_header,
            glossary_entries,
            parsed_subtitles,
        )
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
    async with LLMAPI(
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        max_retries=args.max_retries,
        timeout=args.timeout,
        initial_delay=args.initial_delay,
        max_concurrent=args.max_concurrent,
        provider_order=(
            [p.strip() for p in args.provider.split(",") if p.strip()]
            if args.provider
            else None
        ),
        no_reasoning=args.no_reasoning,
    ) as llm_api:
        subtitle_translator = SubtitleTranslator(
            llm_api=llm_api,
            tag_mode=args.tag_mode,
            filter_bad_words=args.filter_bad_words,
            karaoke_policy=args.karaoke_policy,
        )

        try:
            # Process single file
            if input_path.is_file():
                if not input_path.name.endswith(".srt"):
                    logger.error(f"Not .srt file: {input_path}")
                    return 1
                success = await process_file(str(input_path), subtitle_translator, args)
                return 0 if success else 1

            # Process all .srt files in directory
            elif input_path.is_dir():
                srt_files = list(input_path.glob("**/*.srt"))
                if not srt_files:
                    logger.error(f"No .srt files found in {input_path}")
                    return 1

                logger.info(f"Found {len(srt_files)} .srt files to process")
                failed_files = []

                for srt_file in srt_files:
                    success = await process_file(
                        str(srt_file), subtitle_translator, args
                    )
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
    exit_code = asyncio.run(main())
    sys.exit(exit_code or 0)
