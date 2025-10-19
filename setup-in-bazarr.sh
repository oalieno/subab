#!/bin/sh

# This script is intended to be used in Bazarr's Custom Post-Processing section.
# Prepare a persistent venv in /config/venv and ensure required deps.
# Does NOT execute subab.py; intended to be run as:
#   /config/setup-in-bazarr.sh && /config/subab.py "...args..."

set -eu

VENV_PATH="/config/venv"
PYTHON_BIN="$VENV_PATH/bin/python"
PIP_BIN="$VENV_PATH/bin/pip"

# Create venv if not present
if [ ! -f "$PYTHON_BIN" ]; then
  python3 -m venv "$VENV_PATH"
fi

# Compute missing/mismatched requirements
NEED=""
REQS="httpx==0.28.1 srt==3.5.3 json-repair==0.52.0"
for spec in $REQS; do
  name="${spec%%==*}"
  want="${spec##*==}"
  installed_version="$($PIP_BIN show "$name" 2>/dev/null | awk -F": " '/^Version: /{print $2}')"
  if [ "$installed_version" != "$want" ]; then
    NEED="$NEED $spec"
  fi
done

if [ -n "$NEED" ]; then
  $PIP_BIN install --no-cache-dir -q $NEED
fi

exit 0


