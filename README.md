# SubAB

SubAB is a lightweight CLI tool for translating `.srt` subtitle files using any OpenAI-compatible LLM API. It delivers efficient, high-quality translations right from your terminal, serving as a minimal alternative to heavier tools like [llm-subtrans](https://github.com/machinewrapped/llm-subtrans).

## Key Features

- **1:1 Mapping Enforcement**: Per-line ID tagging (opaque or numeric) guarantees strict input→output alignment, preventing merges/splits (see details below).
- **Adaptive Batching**: Robust retry/downsize when context is too large, automatically splits batches to fit model/output limits and resolve anomalies (see details below).
- **Bazarr Friendly**: Easily integrates with Bazarr for automated subtitle translation. (just a simple script)

### 1:1 Mapping Enforcement (ID Tagging)

LLMs sometimes merge or split lines. To guarantee a strict 1:1 mapping, SubAB adds a short ID token in front of every input line and requires the model to copy it back unchanged. This lets us align outputs to inputs reliably.

- **Tag modes** (choose with `--tag-mode`):
  - `opaque` (default): short random tokens per line (e.g., `aa11|`), unique within a request.
  - `numeric`: simple 0-based numbering (e.g., `0000|`).
- **Local repair**: If the model returns exactly one fewer line (typical merge), SubAB retries only the local neighbors to repair quickly. Otherwise, it falls back to adaptive batching.

### Adaptive Batching

With ID tagging, most 1:1 mapping issues are prevented up front. Adaptive batching acts as a robust retry/downsize mechanism when the model/provider cannot reliably handle the current context size. Typical triggers:

- Max output/token limit is insufficient to return all lines
- Smaller models still fail 1:1 even with tags
- Provider anomalies on large I/O (e.g., blank responses); shrinking the input resolves them

SubAB starts with an optimistic, larger batch for better context. On failure, it splits the batch into two halves and retries, recursively finding the largest batch size the model can handle reliably. Combined with the local-repair step (for the "exactly-one-line missing" special case), this keeps results stable while minimizing retries.

The diagram below illustrates this fallback process:

```
                     [ A Batch of 128 Subtitle Lines ]
                                    |
                          (LLM Translation Fails)
                                    |
                      +-------------+-------------+
                      |                           |
             [ Batch A: 64 Lines ]       [ Batch B: 64 Lines ]
                      |                           |
                (Fails Again)                (Succeeds!)
                      |                           |
          +-----------+-----------+               |
          |                       |               +-------> [ Translated B ]
[ Batch C: 32 Lines ]    [ Batch D: 32 Lines ]
          |                       |
     (Succeeds!)             (Succeeds!)
          |                       |
          |                       |
   [ Translated C ]        [ Translated D ]


Final Result: [ Translated C ] + [ Translated D ] + [ Translated B ]
```

This process primarily downsizes to a reliable context window and serves as a strong retry strategy when limits or provider issues are hit.

### Karaoke Handling

Some providers output karaoke-style, per-syllable lines (e.g., many tiny lines during OP/ED). SubAB can automatically detect these dense short-line segments and apply a policy before batching, reducing noise and token usage. See example below:

```
30
00:00:52,137 --> 00:00:53,388
mi

31
00:00:52,137 --> 00:00:53,528
no

32
00:00:52,137 --> 00:00:53,528
no

33
00:00:52,137 --> 00:00:53,689
sei

34
00:00:52,137 --> 00:00:53,689
sei
```

- Policies (via `--karaoke-policy {remove,skip,translate}`):

  - remove (default): drop karaoke lines; only regular lines are translated.
  - skip: keep karaoke lines unchanged; translate regular lines only.
  - translate: translate everything (may use more tokens).

- Examples:
  - Remove karaoke (default): `python subab.py movie.srt`
  - Keep karaoke but don’t translate it: `python subab.py movie.srt --karaoke-policy skip`
  - Translate everything (including karaoke): `python subab.py movie.srt --karaoke-policy translate`

## Requirements

- Python 3.8 or higher
- `httpx`, `srt`, `json-repair`

## Installation

```bash
uv sync
```

## Usage

### Command-Line Arguments

```
usage: subab.py [-h] --api-base API_BASE --api-key API_KEY [--target-language TARGET_LANGUAGE]
                [--target-lang-code TARGET_LANG_CODE] [--also-skip [ALSO_SKIP ...]] --model MODEL
                [--max-batch-size MAX_BATCH_SIZE] [--max-retries MAX_RETRIES]
                [--initial-delay INITIAL_DELAY] [--max-concurrent MAX_CONCURRENT] [--timeout TIMEOUT]
                [--tag-mode {opaque,numeric}] [--force] [--filter-bad-words]
                [--karaoke-policy {skip,remove,translate}] [--no-header]
                [--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}] input_path

Translate SRT subtitles using an OpenAI-compatible API.

positional arguments:
  input_path            Input SRT file path or folder containing SRT files

options:
  -h, --help            show this help message and exit
  --api-base API_BASE   OpenAI API proxy server base URL
  --api-key API_KEY     API key for authentication
  --target-language TARGET_LANGUAGE
                        The target language for translation (e.g., 'English', 'Japanese', 'Traditional Chinese') (default: Traditional Chinese)
  --target-lang-code TARGET_LANG_CODE
                        The target language code for the output filename (e.g., 'en', 'ja', 'zh-TW') (default: zh-TW)
  --also-skip ALSO_SKIP [ALSO_SKIP ...]
                        Also skip translating if files with these language codes exist (e.g., for target 'zh-TW', you might want to skip if 'zh' exists). (default: None)
  --model MODEL         LLM model
  --max-batch-size MAX_BATCH_SIZE
                        Maximum number of subtitles to translate in one batch (default: 400)
  --max-retries MAX_RETRIES
                        Maximum number of retries for failed requests (default: 3)
  --initial-delay INITIAL_DELAY
                        Initial delay time in seconds for retries (will be doubled each retry) (default: 30.0)
  --max-concurrent MAX_CONCURRENT
                        Maximum number of concurrent API requests (default: 5)
  --timeout TIMEOUT     Timeout in seconds for API requests (default: 60.0)
  --tag-mode {opaque,numeric}
                        Token tagging scheme to enforce 1:1 mapping (default: opaque)
  --force               Force translation even if translated version exists
  --filter-bad-words    Filter out bad words in subtitles
  --karaoke-policy {skip,remove,translate}
                        How to handle karaoke-like short lines (skip=keep as-is, remove=drop, translate=translate) (default: remove)
  --no-header           Do not add a header with translation info to the SRT file
  --log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        Set the logging level (default: INFO)
```

### Examples

#### Translating a Single File

This will translate `movie.en.srt` to Japanese and save it as `movie.ja.srt`. Be sure to specify your API base, API key and model.

```bash
python subab.py "/path/to/your/movie.en.srt" \
    --api-base "https://your.openai.proxy/v1" \
    --api-key "YOUR_API_KEY" \
    --model "your-model-name" \
    --target-language "Japanese" \
    --target-lang-code "ja"
```

#### Translating an Entire Folder

This command will find all `.srt` files in the `/path/to/subs` directory and its subdirectories, and translate them into Traditional Chinese.

```bash
python subab.py "/path/to/subs" \
    --api-base "https://your.openai.proxy/v1" \
    --api-key "YOUR_API_KEY" \
    --model "your-model-name" \
    --target-language "Traditional Chinese" \
    --target-lang-code "zh-TW" \
    --also-skip "zh"
```

### Model Selection

The notes below are based on tests via OpenRouter; limits and behavior can vary by provider and model. Adjust `--max-batch-size` to stay within each model’s effective output/context limits.

- gemini-2.0-flash: best price/performance, but Max Output ~8.2k tokens. Recommended `--max-batch-size` ≈ 400.
- gemini-2.5-flash: higher quality; can usually handle very large batches. `--max-batch-size` ≥ 1024 often finishes in a single batch.
- grok-4-fast: similar behavior to the above; `--max-batch-size` ≥ 1024 typically 1 batch.
- gemma3-12b: minimum viable; recommend `--max-batch-size` ≈ 256 (may trigger local-repair/adaptive batching more often).

Tip: ID tagging guarantees 1:1 structure, and blank/flaky outputs are automatically handled by adaptive batching. However, for the best efficiency with a specific model and provider, it’s still important to adjust `--max-batch-size` (and `--max-concurrent`) to minimize retries and control token usage.

## Integration with [Bazarr](https://github.com/morpheus65535/bazarr)

The most reliable way to use this script with Bazarr is to let the command itself handle the dependency installation within Bazarr's persistent `/config` directory. This avoids the need to modify the Bazarr container image.

### Setup

1.  Place your `subab.py` script inside your Bazarr's `/config` directory. For example: `/path/to/your/bazarr/config/subab.py`.
2.  Go to your Bazarr `Settings` -> `Subtitles`.
3.  Scroll down to the `Custom Post-Processing` section and enable it.
4.  Paste the following one-line command into the `Command` field.

### Command

This command automatically creates a Python virtual environment, installs `httpx` if it's not already present, and then executes the script.

```bash
/bin/sh -c 'VENV_PATH="/config/venv"; if [ ! -f "$VENV_PATH/bin/python" ]; then python3 -m venv "$VENV_PATH"; fi; "$VENV_PATH/bin/pip" install --no-cache-dir -q "httpx==0.28.1" "srt==3.5.3" "json-repair==0.52.0"; exec "$VENV_PATH/bin/python" /config/subab.py "{{subtitles}}" --api-base "https://your.openai.proxy/v1" --api-key "YOUR_API_KEY" --model "your-model-name" --target-language "Traditional Chinese" --target-lang-code "zh-TW" --also-skip "zh"'
```

**Before you save, make sure to customize the following parts of the command:**

- `/config/subab.py`: **Update this** to the actual path where you placed the `subab.py` script inside your Bazarr config volume.
- `--api-key`, `--api-base`, `--model`, and language arguments: **Update these** with your actual API credentials and desired translation settings.

This setup is self-contained, persistent across Bazarr container updates, and ensures the script always has the dependencies it needs to run.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
