"""
数据预处理模块。
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.utils.common import flatten_page_records, normalize_text, save_csv, save_json, save_jsonl
from src.utils.logger import logger, setup_logger


class DataPreprocessor:
    """原始爬取数据预处理器。"""

    def __init__(self, data_dir: str = "data", processed_dir: Optional[str] = None, log_dir: str = "logs"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = Path(processed_dir) if processed_dir else self.data_dir / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        setup_logger(log_dir, "preprocess")

    def load_raw_data(self, filepath: Optional[str | Sequence[str]] = None) -> List[Dict[str, Any]]:
        """加载原始 JSON/JSONL 数据。"""
        filepaths: List[Path] = []

        if filepath is None:
            filepaths.extend(sorted(self.raw_dir.glob("*.json")))
            filepaths.extend(sorted(self.raw_dir.glob("*.jsonl")))
        elif isinstance(filepath, (list, tuple)):
            filepaths.extend(Path(item) for item in filepath)
        else:
            target = Path(filepath)
            if target.is_dir():
                filepaths.extend(sorted(target.glob("*.json")))
                filepaths.extend(sorted(target.glob("*.jsonl")))
            else:
                filepaths.append(target)

        all_data: List[Dict[str, Any]] = []
        for file in filepaths:
            if not file.exists():
                logger.warning("输入文件不存在: {}", file)
                continue

            logger.info("加载文件: {}", file)
            if file.suffix == ".jsonl":
                with file.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if line:
                            all_data.append(json.loads(line))
            else:
                with file.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                    if isinstance(payload, list):
                        all_data.extend(payload)
                    elif isinstance(payload, dict):
                        all_data.append(payload)

        logger.info("共加载 {} 条原始记录", len(all_data))
        return all_data

    def flatten_records(self, data: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """将页面级记录转成评论/回复级记录。"""
        if not data:
            return []

        if data and "post" in data[0] and "comments" in data[0]:
            flat = flatten_page_records(data)
        else:
            flat = [dict(item) for item in data]

        logger.info("扁平化后得到 {} 条评论/回复记录", len(flat))
        return flat

    def clean_text(self, text: str) -> str:
        if not text:
            return ""

        cleaned = normalize_text(text)
        cleaned = re.sub(r"@\S+", "", cleaned)
        cleaned = re.sub(r"#([^#]+)#", r"\1", cleaned)
        cleaned = re.sub(r"[^\w\s\u4e00-\u9fff，。！？、；：”“‘’（）【】《》…,.!?;:()\\-]", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def filter_comments(
        self,
        data: Sequence[Dict[str, Any]],
        min_length: int = 10,
        max_length: int = 512,
        min_likes: int = 0,
    ) -> List[Dict[str, Any]]:
        """按文本长度和互动量过滤。"""
        filtered: List[Dict[str, Any]] = []

        for item in data:
            record = dict(item)
            text = record.get("text") or record.get("content") or record.get("comment_content") or ""
            text = self.clean_text(text)
            if not text:
                continue
            if len(text) < min_length or len(text) > max_length:
                continue
            if int(record.get("likes", 0) or 0) < min_likes:
                continue

            record["text"] = text
            record["content"] = text
            record["dialogue"] = [self.clean_text(part) for part in record.get("dialogue", []) if self.clean_text(part)]
            filtered.append(record)

        logger.info("过滤后剩余 {}/{} 条记录", len(filtered), len(data))
        return filtered

    def extract_dialogue_pairs(
        self,
        data: Sequence[Dict[str, Any]],
        min_reply_length: int = 5,
    ) -> List[Dict[str, Any]]:
        """提取统一格式对话样本。"""
        samples: List[Dict[str, Any]] = []

        for item in data:
            dialogue = [self.clean_text(part) for part in item.get("dialogue", []) if self.clean_text(part)]
            if not dialogue:
                single = self.clean_text(item.get("text") or item.get("content") or "")
                if not single:
                    continue
                dialogue = [single]

            if len(dialogue) > 1 and len(dialogue[-1]) < min_reply_length:
                continue

            sample = {
                "platform": item.get("platform", "unknown"),
                "source_url": item.get("source_url", ""),
                "post_id": item.get("post_id", ""),
                "dialogue": dialogue,
                "text": " [SEP] ".join(dialogue),
                "content": dialogue[-1],
                "likes": int(item.get("likes", 0) or 0),
                "collects": int(item.get("collects", 0) or 0),
                "comments": int(item.get("comments", 0) or 0),
                "views": int(item.get("views", 0) or 0),
                "level": int(item.get("level", 1) or 1),
                "comment_user": item.get("comment_user", ""),
                "reply_user": item.get("reply_user", ""),
                "post_title": item.get("post_title", ""),
                "post_content": item.get("post_content", ""),
                "post_author": item.get("post_author", ""),
                "post_likes": int(item.get("post_likes", 0) or 0),
                "post_collects": int(item.get("post_collects", 0) or 0),
                "post_comments": int(item.get("post_comments", 0) or 0),
                "post_views": int(item.get("post_views", 0) or 0),
            }
            samples.append(sample)

        logger.info("提取对话样本 {} 条", len(samples))
        return samples

    def split_dataset(
        self,
        data: Sequence[Dict[str, Any]],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        shuffle: bool = True,
        seed: int = 42,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """划分训练/验证/测试集。"""
        if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
            raise ValueError("train_ratio + val_ratio + test_ratio 必须等于 1")

        items = list(data)
        if shuffle:
            random.Random(seed).shuffle(items)

        total = len(items)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        train_data = items[:train_end]
        val_data = items[train_end:val_end]
        test_data = items[val_end:]

        logger.info("数据集划分: train={}, val={}, test={}", len(train_data), len(val_data), len(test_data))
        return train_data, val_data, test_data

    def save_dataset(
        self,
        train_data: Sequence[Dict[str, Any]],
        val_data: Sequence[Dict[str, Any]],
        test_data: Sequence[Dict[str, Any]],
        prefix: str = "dataset",
        save_csv_copy: bool = True,
    ) -> Dict[str, str]:
        """保存数据集。"""
        paths = {
            "train_jsonl": save_jsonl(train_data, self.processed_dir / f"{prefix}_train.jsonl"),
            "val_jsonl": save_jsonl(val_data, self.processed_dir / f"{prefix}_val.jsonl"),
            "test_jsonl": save_jsonl(test_data, self.processed_dir / f"{prefix}_test.jsonl"),
        }

        if save_csv_copy:
            paths["train_csv"] = save_csv(list(train_data), self.processed_dir / f"{prefix}_train.csv")
            paths["val_csv"] = save_csv(list(val_data), self.processed_dir / f"{prefix}_val.csv")
            paths["test_csv"] = save_csv(list(test_data), self.processed_dir / f"{prefix}_test.csv")

        return paths

    def build_statistics(
        self,
        train: Sequence[Dict[str, Any]],
        val: Sequence[Dict[str, Any]],
        test: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        all_data = list(train) + list(val) + list(test)
        lengths = [len(item.get("text", "")) for item in all_data if item.get("text")]
        likes = [int(item.get("likes", 0) or 0) for item in all_data]

        return {
            "total": len(all_data),
            "train": len(train),
            "val": len(val),
            "test": len(test),
            "avg_length": round(sum(lengths) / len(lengths), 2) if lengths else 0,
            "min_length": min(lengths) if lengths else 0,
            "max_length": max(lengths) if lengths else 0,
            "avg_likes": round(sum(likes) / len(likes), 2) if likes else 0,
            "max_likes": max(likes) if likes else 0,
        }

    def process_all(
        self,
        input_file: Optional[str | Sequence[str]] = None,
        output_prefix: str = "dataset",
        min_length: int = 10,
        max_length: int = 512,
        min_likes: int = 0,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        shuffle: bool = True,
        labeler: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """执行完整预处理流程。"""
        raw_data = self.load_raw_data(input_file)
        flat_records = self.flatten_records(raw_data)
        filtered = self.filter_comments(flat_records, min_length=min_length, max_length=max_length, min_likes=min_likes)
        dialogues = self.extract_dialogue_pairs(filtered)

        if labeler is not None:
            dialogues = labeler.generate_labels(dialogues)

        train, val, test = self.split_dataset(
            dialogues,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            shuffle=shuffle,
        )
        dataset_paths = self.save_dataset(train, val, test, prefix=output_prefix)
        flat_path = save_jsonl(filtered, self.processed_dir / f"{output_prefix}_flat.jsonl")

        stats = self.build_statistics(train, val, test)
        stats["flat_records"] = len(filtered)
        stats["output_prefix"] = output_prefix
        summary_path = save_json(stats, self.processed_dir / f"{output_prefix}_summary.json")

        logger.info("预处理完成: {}", summary_path)
        return {
            "train": train,
            "val": val,
            "test": test,
            "flat_records": filtered,
            "stats": stats,
            "paths": {
                **dataset_paths,
                "flat_jsonl": flat_path,
                "summary_json": summary_path,
            },
        }
