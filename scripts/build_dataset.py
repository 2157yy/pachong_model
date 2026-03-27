#!/usr/bin/env python3
"""
构建训练数据集。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.utils.labeler import PseudoLabeler
from src.utils.preprocess import DataPreprocessor


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从原始抓取结果构建训练数据集")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--input", nargs="*", help="输入文件或目录，默认读取 data/raw")
    parser.add_argument("--output-prefix", help="输出前缀")
    parser.add_argument("--no-label", action="store_true", help="不生成伪标签")
    return parser.parse_args()


def main() -> int:
    args = build_args()
    config = load_config(args.config)

    pre_cfg = config.get("preprocess", {})
    paths_cfg = config.get("paths", {})
    label_cfg = config.get("labeler", {})

    preprocessor = DataPreprocessor(
        data_dir=paths_cfg.get("data_dir", "data"),
        processed_dir=paths_cfg.get("processed_dir", "data/processed"),
        log_dir=paths_cfg.get("log_dir", "logs"),
    )
    labeler = None if args.no_label else PseudoLabeler(**label_cfg)

    result = preprocessor.process_all(
        input_file=args.input if args.input else None,
        output_prefix=args.output_prefix or pre_cfg.get("output_prefix", "dataset"),
        min_length=int(pre_cfg.get("min_length", 10)),
        max_length=int(pre_cfg.get("max_length", 512)),
        min_likes=int(pre_cfg.get("min_likes", 0)),
        train_ratio=float(pre_cfg.get("train_ratio", 0.8)),
        val_ratio=float(pre_cfg.get("val_ratio", 0.1)),
        test_ratio=float(pre_cfg.get("test_ratio", 0.1)),
        shuffle=bool(pre_cfg.get("shuffle", True)),
        labeler=labeler,
    )

    for name, path in result["paths"].items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
