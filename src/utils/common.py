"""
通用数据处理和导出工具。
"""
from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence
from urllib.parse import quote


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def parse_number(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)

    text = str(value).strip().lower().replace(",", "")
    if not text:
        return 0

    multiplier = 1
    if text.endswith(("w", "万")):
        multiplier = 10000
    elif text.endswith(("k", "千")):
        multiplier = 1000
    elif text.endswith("m"):
        multiplier = 1000000

    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return 0
    return int(float(match.group()) * multiplier)


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    cleaned = str(text)
    cleaned = re.sub(r"http[s]?://\S+", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def format_timestamp(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, str) and not value.isdigit():
        return value
    try:
        number = int(value)
    except (TypeError, ValueError):
        return str(value)

    if number <= 0:
        return ""
    if number > 10_000_000_000:
        number = number // 1000
    return datetime.fromtimestamp(number).isoformat(timespec="seconds")


def safe_text_candidates(*values: Any) -> str:
    for value in values:
        text = normalize_text(value)
        if text:
            return text
    return ""


def read_url_inputs(url: str | None = None, url_file: str | None = None) -> List[str]:
    urls: List[str] = []
    if url:
        urls.append(url.strip())
    if url_file:
        with Path(url_file).open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line and not line.startswith("#"):
                    urls.append(line)
    deduped: List[str] = []
    seen = set()
    for item in urls:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def write_url_file(urls: Sequence[str], path: str | Path) -> str:
    target = Path(path)
    ensure_directory(target.parent)
    deduped: List[str] = []
    seen = set()
    for item in urls:
        normalized = str(item).strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(normalized)
    with target.open("w", encoding="utf-8") as file:
        for item in deduped:
            file.write(f"{item}\n")
    return str(target)


def quote_keyword(keyword: str) -> str:
    return quote(keyword, safe="")


def detect_platform(url: str) -> str:
    lowered = url.lower()
    if "douyin.com" in lowered or "iesdouyin.com" in lowered:
        return "douyin"
    if "xiaohongshu.com" in lowered or "xhslink.com" in lowered:
        return "xiaohongshu"
    raise ValueError(f"无法从 URL 识别平台: {url}")


def flatten_page_records(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for record in records:
        post = record.get("post", {}) or {}
        comments = record.get("comments", []) or []
        base = {
            "platform": record.get("platform", "unknown"),
            "source_url": record.get("source_url", ""),
            "crawled_at": record.get("crawled_at", ""),
            "post_id": post.get("post_id", ""),
            "post_title": post.get("title", ""),
            "post_content": post.get("content", ""),
            "post_author": post.get("author", ""),
            "post_likes": parse_number(post.get("likes", 0)),
            "post_collects": parse_number(post.get("collects", 0)),
            "post_comments": parse_number(post.get("comments", 0)),
            "post_shares": parse_number(post.get("shares", 0)),
            "post_views": parse_number(post.get("views", 0)),
            "post_publish_time": post.get("publish_time", ""),
        }

        for comment in comments:
            comment_row = {
                **base,
                "level": 1,
                "comment_id": comment.get("comment_id", ""),
                "comment_user": comment.get("user", ""),
                "comment_content": comment.get("content", ""),
                "comment_likes": parse_number(comment.get("likes", 0)),
                "comment_time": comment.get("time", ""),
                "reply_count": parse_number(comment.get("reply_count", 0)),
                "reply_id": "",
                "reply_user": "",
                "reply_content": "",
                "reply_likes": 0,
                "reply_time": "",
                "dialogue": [normalize_text(comment.get("content", ""))],
                "text": normalize_text(comment.get("content", "")),
                "content": normalize_text(comment.get("content", "")),
                "likes": parse_number(comment.get("likes", 0)),
                "collects": parse_number(post.get("collects", 0)),
                "comments": parse_number(post.get("comments", 0)),
                "views": parse_number(post.get("views", 0)),
            }
            rows.append(comment_row)

            for reply in comment.get("replies", []) or []:
                dialogue = [
                    normalize_text(comment.get("content", "")),
                    normalize_text(reply.get("content", "")),
                ]
                reply_row = {
                    **base,
                    "level": 2,
                    "comment_id": comment.get("comment_id", ""),
                    "comment_user": comment.get("user", ""),
                    "comment_content": comment.get("content", ""),
                    "comment_likes": parse_number(comment.get("likes", 0)),
                    "comment_time": comment.get("time", ""),
                    "reply_count": parse_number(comment.get("reply_count", 0)),
                    "reply_id": reply.get("reply_id", ""),
                    "reply_user": reply.get("user", ""),
                    "reply_content": reply.get("content", ""),
                    "reply_likes": parse_number(reply.get("likes", 0)),
                    "reply_time": reply.get("time", ""),
                    "dialogue": dialogue,
                    "text": " [SEP] ".join([part for part in dialogue if part]),
                    "content": normalize_text(reply.get("content", "")) or normalize_text(comment.get("content", "")),
                    "likes": parse_number(reply.get("likes", 0) or comment.get("likes", 0)),
                    "collects": parse_number(post.get("collects", 0)),
                    "comments": parse_number(post.get("comments", 0)),
                    "views": parse_number(post.get("views", 0)),
                }
                rows.append(reply_row)

    return rows


def summarize_records(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    flat_rows = flatten_page_records(records)
    return {
        "record_count": len(records),
        "flat_row_count": len(flat_rows),
        "platforms": sorted({record.get("platform", "unknown") for record in records}),
    }


def save_json(data: Any, path: str | Path) -> str:
    target = Path(path)
    ensure_directory(target.parent)
    with target.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    return str(target)


def save_jsonl(rows: Iterable[Dict[str, Any]], path: str | Path) -> str:
    target = Path(path)
    ensure_directory(target.parent)
    with target.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    return str(target)


def save_csv(rows: Sequence[Dict[str, Any]], path: str | Path) -> str:
    target = Path(path)
    ensure_directory(target.parent)
    if not rows:
        with target.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["empty"])
        return str(target)

    keys = sorted({key for row in rows for key in row.keys()})
    with target.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    return str(target)


def save_records_bundle(
    records: Sequence[Dict[str, Any]],
    output_dir: str | Path,
    basename: str,
    formats: Sequence[str] = ("json", "jsonl", "csv"),
) -> Dict[str, str]:
    output_dir = ensure_directory(output_dir)
    formats = tuple(sorted({item.lower() for item in formats}))
    flat_rows = flatten_page_records(records)
    summary = summarize_records(records)

    paths: Dict[str, str] = {}
    if "json" in formats:
        paths["records_json"] = save_json(list(records), output_dir / f"{basename}_records.json")
        paths["summary_json"] = save_json(summary, output_dir / f"{basename}_summary.json")
    if "jsonl" in formats:
        paths["records_jsonl"] = save_jsonl(records, output_dir / f"{basename}_records.jsonl")
        paths["dialogues_jsonl"] = save_jsonl(flat_rows, output_dir / f"{basename}_dialogues.jsonl")
    if "csv" in formats:
        paths["comments_csv"] = save_csv(flat_rows, output_dir / f"{basename}_comments.csv")

    return paths
