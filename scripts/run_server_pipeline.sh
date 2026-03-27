#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG_FILE="${1:-configs/config.server.yaml}"
KEYWORD="${2:-自嘲熊}"
URL_FILE="${3:-urls.txt}"

python3 scripts/run_crawler.py \
  --config "$CONFIG_FILE" \
  --platform auto \
  --keyword "$KEYWORD" \
  --save-url-file "$URL_FILE"

python3 scripts/build_dataset.py \
  --config "$CONFIG_FILE"

python3 scripts/train_classifier.py \
  --config "$CONFIG_FILE" \
  --train-file data/processed/dataset_train.jsonl \
  --val-file data/processed/dataset_val.jsonl \
  --test-file data/processed/dataset_test.jsonl

echo "pipeline done"
