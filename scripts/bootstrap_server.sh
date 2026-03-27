#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

python3 -m pip install -r requirements.txt
python3 -m playwright install chromium

mkdir -p \
  data/raw \
  data/processed \
  data/browser_state \
  data/labels \
  checkpoints \
  logs

echo "server bootstrap done"
