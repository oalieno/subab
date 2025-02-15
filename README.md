# SRTGPT for Bazarr

A small script that can translate `.srt` subtitles using **Modern Large Language Model**. Specifically tuned for **Bazarr Custom Post-Processing** purpose.

## Setup for Bazarr Custom Post-Processing

Paste the below line into your Bazarr custom post-processing section.

```bash
apk info py3-httpx >/dev/null 2>&1 || apk add py3-httpx && python /config/srtgpt.py "{{subtitles}}"
```
