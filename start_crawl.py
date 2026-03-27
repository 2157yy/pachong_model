#!/usr/bin/env python3
"""
VSCode 可直接点击运行的抓取入口。

运行前请确保 VSCode 选中的 Python 解释器就是你的 conda 环境。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

# 按需修改这里的默认参数
CONFIG_FILE = ROOT / "configs" / "config.server.yaml"
KEYWORD = "自嘲熊"
URL_FILE = ROOT / "urls.txt"
MAX_COMMENTS = 300
SCROLL_TIMES = 15


def main() -> int:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_crawler.py"),
        "--config",
        str(CONFIG_FILE),
        "--platform",
        "auto",
        "--keyword",
        KEYWORD,
        "--save-url-file",
        str(URL_FILE),
        "--max-comments",
        str(MAX_COMMENTS),
        "--scroll-times",
        str(SCROLL_TIMES),
    ]

    return subprocess.run(command, cwd=str(ROOT), check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
