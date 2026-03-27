#!/usr/bin/env python3
"""
训练质量分类模型。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model.trainer import Trainer
from src.utils.config import load_config


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练质量分类模型")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--train-file", required=True, help="训练集 JSONL")
    parser.add_argument("--val-file", required=True, help="验证集 JSONL")
    parser.add_argument("--test-file", help="测试集 JSONL")
    parser.add_argument("--save-name", help="模型输出目录名")
    parser.add_argument("--backend", choices=["auto", "paddle", "sklearn"], help="训练后端")
    return parser.parse_args()


def main() -> int:
    args = build_args()
    config = load_config(args.config)
    model_cfg = config.get("model", {})
    paths_cfg = config.get("paths", {})

    trainer = Trainer(
        model_name=model_cfg.get("name", "ernie-3.0-base"),
        num_labels=int(model_cfg.get("num_labels", 3)),
        max_length=int(model_cfg.get("max_length", 512)),
        learning_rate=float(model_cfg.get("learning_rate", 2e-5)),
        batch_size=int(model_cfg.get("batch_size", 32)),
        epochs=int(model_cfg.get("epochs", 5)),
        warmup_ratio=float(model_cfg.get("warmup_ratio", 0.1)),
        device=model_cfg.get("device", "cpu"),
        output_dir=paths_cfg.get("checkpoint_dir", "checkpoints"),
        backend=args.backend or model_cfg.get("backend", "auto"),
    )

    train_data = trainer.load_jsonl(args.train_file)
    val_data = trainer.load_jsonl(args.val_file)
    test_data = trainer.load_jsonl(args.test_file) if args.test_file else None

    history = trainer.train(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        save_name=args.save_name or model_cfg.get("save_name", "best_model"),
    )
    print(history)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
