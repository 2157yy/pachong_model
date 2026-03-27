"""
伪标签生成模块。
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

from src.utils.logger import logger


class PseudoLabeler:
    """根据互动特征生成 high/medium/low 标签。"""

    LABEL_MAP = {"low": 0, "medium": 1, "high": 2}

    def __init__(
        self,
        like_threshold_high: int = 1000,
        like_threshold_low: int = 100,
        collect_threshold_high: int = 500,
        collect_threshold_low: int = 50,
        comment_threshold_high: int = 100,
        comment_threshold_low: int = 20,
        content_length_high: int = 200,
        content_length_low: int = 50,
        engagement_rate_high: float = 0.05,
        engagement_rate_low: float = 0.01,
    ):
        self.like_threshold_high = like_threshold_high
        self.like_threshold_low = like_threshold_low
        self.collect_threshold_high = collect_threshold_high
        self.collect_threshold_low = collect_threshold_low
        self.comment_threshold_high = comment_threshold_high
        self.comment_threshold_low = comment_threshold_low
        self.content_length_high = content_length_high
        self.content_length_low = content_length_low
        self.engagement_rate_high = engagement_rate_high
        self.engagement_rate_low = engagement_rate_low

    def generate_label(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """生成标签及打分细节。"""
        likes = int(item.get("likes", 0) or item.get("post_likes", 0) or 0)
        collects = int(item.get("collects", 0) or item.get("post_collects", 0) or 0)
        comments = int(item.get("comments", 0) or item.get("post_comments", 0) or 0)
        views = int(item.get("views", 0) or item.get("post_views", 0) or 1)
        text = item.get("text") or " ".join(item.get("dialogue", [])) or item.get("content", "")

        score = 0
        reasons: List[str] = []

        if likes > self.like_threshold_high:
            score += 3
            reasons.append("likes_high")
        elif likes > self.like_threshold_low:
            score += 2
            reasons.append("likes_medium")
        elif likes > 10:
            score += 1
            reasons.append("likes_low")

        if collects > self.collect_threshold_high:
            score += 2
            reasons.append("collects_high")
        elif collects > self.collect_threshold_low:
            score += 1
            reasons.append("collects_medium")

        if comments > self.comment_threshold_high:
            score += 2
            reasons.append("comments_high")
        elif comments > self.comment_threshold_low:
            score += 1
            reasons.append("comments_medium")

        length = len(text or "")
        if length > self.content_length_high:
            score += 2
            reasons.append("length_high")
        elif length > self.content_length_low:
            score += 1
            reasons.append("length_medium")

        engagement_rate = (likes + collects) / max(views, 1)
        if engagement_rate > self.engagement_rate_high:
            score += 2
            reasons.append("engagement_high")
        elif engagement_rate > self.engagement_rate_low:
            score += 1
            reasons.append("engagement_medium")

        if score >= 8:
            label = "high"
        elif score >= 5:
            label = "medium"
        else:
            label = "low"

        return {
            "label": label,
            "label_id": self.LABEL_MAP[label],
            "label_score": score,
            "label_reasons": reasons,
        }

    def generate_labels(self, data: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        labeled_data: List[Dict[str, Any]] = []
        counter = {"high": 0, "medium": 0, "low": 0}

        for item in data:
            labeled = dict(item)
            labeled.update(self.generate_label(labeled))
            counter[labeled["label"]] += 1
            labeled_data.append(labeled)

        total = len(labeled_data)
        if total:
            logger.info(
                "标签生成完成: total={}, high={}, medium={}, low={}",
                total,
                counter["high"],
                counter["medium"],
                counter["low"],
            )
        return labeled_data

    def analyze_distribution(self, data: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        stats = {"high": [], "medium": [], "low": []}
        for item in data:
            stats.setdefault(item.get("label", "low"), []).append(int(item.get("likes", 0) or 0))

        total = len(data)
        return {
            "total": total,
            "distribution": {
                label: {
                    "count": len(values),
                    "percentage": round(len(values) / total * 100, 2) if total else 0,
                    "avg_likes": round(sum(values) / len(values), 2) if values else 0,
                }
                for label, values in stats.items()
            },
        }
