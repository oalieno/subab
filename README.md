# SubAB

SubAB (Subtitle Adaptive Batching) is a lightweight CLI tool for translating `.srt` subtitle files using any OpenAI-compatible LLM API. It delivers efficient, high-quality translations right from your terminal, serving as a minimal alternative to heavier tools like [llm-subtrans](https://github.com/machinewrapped/llm-subtrans).

## Key Features

- **Adaptive Batching**: Enables efficient, context-aware batch translations by automatically splitting and retrying oversized batches, so you don't have to worry about choosing the perfect batch size (though very large batches may increase token usage on repeated failures). See how it works below.
- **Minimal Dependencies**: Requires only the `httpx` library for API requests, ensuring a lightweight footprint and easy setup.
- **Bazarr Friendly**: Easily integrates with Bazarr for automated subtitle translation. (just a simple script)

### How Adaptive Batching Works

**The Challenge:** Translating subtitles line-by-line is fast but often produces low-quality, out-of-context translations. Sending a large batch of lines at once gives the LLM crucial context, resulting in much better translations. However, the larger the batch, the more likely an LLM is to fail at maintaining a strict 1-to-1 line mapping, sometimes returning 30 or maybe 33 lines for a 32-line input, which breaks the subtitle file.

**The Solution:** SubAB tackles this problem by starting with an optimistic, large batch size to maximize context and translation quality. If the LLM fails to return the correct number of lines, the script doesn't quit. Instead, it "adaptively" splits the batch into two smaller halves and retries each one. This recursive process finds the largest possible batch size that the LLM can handle reliably, ensuring you get the best possible translation quality without sacrificing the structural integrity of the SRT file.

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

This process ensures maximum context for better translations while automatically handling failures by reducing batch sizes.

## Requirements

- Python 3.8 or higher
- httpx library (installed via uv or pip)

## Installation

```bash
uv sync
```

Alternatively, install with `pip install httpx` and run the script directly.

## Usage

### Command-Line Arguments

```
usage: subab.py [-h] --api-key API_KEY [--api-base API_BASE] [--target-language TARGET_LANGUAGE]
                 [--target-lang-code TARGET_LANG_CODE] [--model MODEL] [--max-batch-size MAX_BATCH_SIZE]
                 [--max-retries MAX_RETRIES] [--initial-delay INITIAL_DELAY]
                 [--max-concurrent MAX_CONCURRENT] [--timeout TIMEOUT] [--force]
                 [--filter-bad-words] [--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                 input_path

Translate SRT subtitles using an OpenAI-compatible API.

positional arguments:
  input_path            Input SRT file path or folder containing SRT files

options:
  -h, --help            show this help message and exit
  --api-key API_KEY     API key for authentication
  --api-base API_BASE   OpenAI API proxy server base URL
  --target-language TARGET_LANGUAGE
                        The target language for translation (e.g., 'English', 'Japanese', 'Traditional Chinese') (default: Traditional Chinese)
  --target-lang-code TARGET_LANG_CODE
                        The target language code for the output filename (e.g., 'en', 'ja', 'zh-TW') (default: zh-TW)
  --also-skip ALSO_SKIP [ALSO_SKIP ...]
                        Also skip translating if files with these language codes exist (e.g., for target 'zh-TW', you might want to skip if 'zh' exists). (default: None)
  --model MODEL         LLM model
  --max-batch-size MAX_BATCH_SIZE
                        Maximum number of subtitles to translate in one batch (default: 128)
  --max-retries MAX_RETRIES
                        Maximum number of retries for failed requests (default: 3)
  --initial-delay INITIAL_DELAY
                        Initial delay time in seconds for retries (will be doubled each retry) (default: 1.0)
  --max-concurrent MAX_CONCURRENT
                        Maximum number of concurrent API requests (default: 5)
  --timeout TIMEOUT     Timeout in seconds for API requests (default: 60.0)
  --force               Force translation even if translated version exists
  --filter-bad-words    Filter out bad words in subtitles
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
/bin/sh -c 'VENV_PATH="/config/venv"; if [ ! -f "$VENV_PATH/bin/python" ]; then python3 -m venv "$VENV_PATH"; fi; "$VENV_PATH/bin/pip" install --no-cache-dir -q httpx; exec "$VENV_PATH/bin/python" /config/subab.py "{{subtitles}}" --api-base "https://your.openai.proxy/v1" --api-key "YOUR_API_KEY" --model "your-model-name" --target-language "Traditional Chinese" --target-lang-code "zh-TW" --also-skip "zh"'
```

**Before you save, make sure to customize the following parts of the command:**
-   `/config/subab.py`: **Update this** to the actual path where you placed the `subab.py` script inside your Bazarr config volume.
-   `--api-key`, `--api-base`, `--model`, and language arguments: **Update these** with your actual API credentials and desired translation settings.

This setup is self-contained, persistent across Bazarr container updates, and ensures the script always has the dependencies it needs to run.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
