#!/usr/bin/env python3
"""
一键抓取抖音/小红书评论数据。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.crawler.douyin import DouyinCrawler
from src.crawler.xiaohongshu import XiaohongshuCrawler
from src.utils.common import detect_platform, read_url_inputs
from src.utils.config import deep_update, load_config
from src.utils.logger import logger


CRAWLER_MAP = {
    "douyin": DouyinCrawler,
    "xiaohongshu": XiaohongshuCrawler,
}


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="抓取抖音/小红书评论数据")
    parser.add_argument("--config", default="configs/config.yaml", help="配置文件路径")
    parser.add_argument("--platform", choices=["auto", "douyin", "xiaohongshu"], default="auto", help="平台类型")
    parser.add_argument("--url", help="单个目标链接")
    parser.add_argument("--url-file", help="批量 URL 文件，每行一个")
    parser.add_argument("--headless", action="store_true", help="启用无头模式")
    parser.add_argument("--max-comments", type=int, help="单链接最大评论数")
    parser.add_argument("--scroll-times", type=int, help="滚动次数")
    parser.add_argument("--data-dir", help="输出目录")
    parser.add_argument("--output-prefix", help="输出文件前缀")
    parser.add_argument("--storage-state", help="浏览器登录态文件")
    parser.add_argument("--save-storage-state", help="结束后保存登录态到此文件")
    parser.add_argument("--cookies-file", help="cookies JSON 文件")
    parser.add_argument("--proxy", help="代理地址")
    parser.add_argument("--formats", nargs="+", choices=["json", "jsonl", "csv"], help="导出格式")
    return parser.parse_args()


def resolve_platform_groups(urls: List[str], platform: str) -> Dict[str, List[str]]:
    if platform != "auto":
        return {platform: urls}

    groups: Dict[str, List[str]] = {}
    for url in urls:
        detected = detect_platform(url)
        groups.setdefault(detected, []).append(url)
    return groups


def main() -> int:
    args = build_args()
    config = load_config(args.config)

    urls = read_url_inputs(args.url, args.url_file)
    if not urls:
        raise SystemExit("请通过 --url 或 --url-file 提供目标链接")

    overrides = {
        "crawler": {
            "headless": args.headless if args.headless else config.get("crawler", {}).get("headless", True),
        }
    }
    if args.max_comments is not None:
        overrides["crawler"]["max_comments"] = args.max_comments
    if args.scroll_times is not None:
        overrides["crawler"]["scroll_times"] = args.scroll_times
    if args.data_dir:
        overrides["paths"] = {"raw_dir": args.data_dir}
    if args.proxy:
        overrides["crawler"]["proxy"] = args.proxy
    config = deep_update(config, overrides)

    crawler_config = config.get("crawler", {})
    paths_config = config.get("paths", {})
    formats = args.formats or crawler_config.get("export_formats", ["json", "jsonl", "csv"])

    groups = resolve_platform_groups(urls, args.platform)
    all_paths: Dict[str, str] = {}

    for platform, group_urls in groups.items():
        crawler_cls = CRAWLER_MAP[platform]
        platform_config = crawler_config.get(platform, {})
        raw_dir = paths_config.get("raw_dir", "data/raw")
        crawler = crawler_cls(
            headless=bool(crawler_config.get("headless", True)),
            proxy=crawler_config.get("proxy"),
            user_agent=crawler_config.get("user_agent"),
            data_dir=raw_dir,
            log_dir=paths_config.get("log_dir", "logs"),
            request_delay=float(crawler_config.get("request_delay", 2.0)),
            request_delay_jitter=float(crawler_config.get("request_delay_jitter", 1.0)),
            navigation_timeout_ms=int(crawler_config.get("navigation_timeout_ms", 45000)),
            locale=crawler_config.get("locale", "zh-CN"),
            timezone_id=crawler_config.get("timezone_id", "Asia/Shanghai"),
            storage_state=args.storage_state or platform_config.get("storage_state"),
            save_storage_state=args.save_storage_state or platform_config.get("save_storage_state"),
            cookies_file=args.cookies_file or platform_config.get("cookies_file"),
            viewport_width=int(crawler_config.get("viewport_width", 1440)),
            viewport_height=int(crawler_config.get("viewport_height", 960)),
            log_level=config.get("logging", {}).get("level", "INFO"),
        )

        with crawler:
            records = crawler.crawl_multiple(
                group_urls,
                max_comments=int(crawler_config.get("max_comments", 100)),
                scroll_times=int(crawler_config.get("scroll_times", 8)),
            )
            prefix = args.output_prefix or platform_config.get("output_prefix", platform)
            all_paths.update(crawler.export_records(records, output_prefix=prefix, formats=formats))

    for name, path in all_paths.items():
        logger.info("{} => {}", name, path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
