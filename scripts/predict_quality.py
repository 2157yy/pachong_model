#!/usr/bin/env python3
"""
对文本或已抓取记录做质量预测/排序。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model.classifier import QualityClassifier
from src.utils.common import flatten_page_records, save_csv, save_json, save_jsonl


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="预测文本或记录质量")
    parser.add_argument("--model-dir", required=True, help="训练后模型目录")
    parser.add_argument("--input-file", required=True, help="输入 JSON/JSONL 文件")
    parser.add_argument("--output-file", required=True, help="输出文件前缀，不带后缀")
    return parser.parse_args()


def load_records(path: Path) -> List[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
        if isinstance(payload, list):
            return payload
        return [payload]


def main() -> int:
    args = build_args()
    classifier = QualityClassifier(model_dir=args.model_dir)
    rows = load_records(Path(args.input_file))
    if rows and "post" in rows[0] and "comments" in rows[0]:
        rows = flatten_page_records(rows)

    texts = [row.get("text") or " [SEP] ".join(row.get("dialogue", [])) or row.get("content", "") for row in rows]
    predictions = classifier.predict_batch(texts)

    scored_rows = []
    for row, (label, confidence) in zip(rows, predictions):
        result = dict(row)
        result["predicted_label"] = label
        result["predicted_confidence"] = round(confidence, 6)
        scored_rows.append(result)

    scored_rows.sort(key=lambda item: item.get("predicted_confidence", 0), reverse=True)

    prefix = Path(args.output_file)
    save_json(scored_rows, prefix.with_suffix(".json"))
    save_jsonl(scored_rows, prefix.with_suffix(".jsonl"))
    save_csv(scored_rows, prefix.with_suffix(".csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
