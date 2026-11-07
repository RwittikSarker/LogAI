#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p .cache/pip .cache/tmp models data logs
export PIP_CACHE_DIR="$PWD/.cache/pip"
export TMPDIR="$PWD/.cache/tmp"

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt

echo
echo "Setup done. Activate with: source .venv/bin/activate"