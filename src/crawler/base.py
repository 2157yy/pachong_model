"""
爬虫基础类。

提供浏览器会话管理、网络响应捕获、统一导出和批量爬取能力。
"""
from __future__ import annotations

import json
import random
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence

from src.utils.common import iso_now, save_records_bundle
from src.utils.logger import logger, setup_logger

if TYPE_CHECKING:
    from playwright.sync_api import Browser, BrowserContext, Page, Playwright, Response
else:
    Browser = BrowserContext = Page = Playwright = Response = Any


class BaseCrawler(ABC):
    """爬虫基类。"""

    platform = "unknown"
    response_patterns: Sequence[str] = ()

    def __init__(
        self,
        headless: bool = True,
        proxy: Optional[str] = None,
        user_agent: Optional[str] = None,
        data_dir: str = "data/raw",
        log_dir: str = "logs",
        request_delay: float = 2.0,
        request_delay_jitter: float = 1.0,
        navigation_timeout_ms: int = 45000,
        locale: str = "zh-CN",
        timezone_id: str = "Asia/Shanghai",
        storage_state: Optional[str] = None,
        save_storage_state: Optional[str] = None,
        cookies_file: Optional[str] = None,
        viewport_width: int = 1440,
        viewport_height: int = 960,
        log_level: str = "INFO",
    ):
        self.headless = headless
        self.proxy = proxy
        self.user_agent = user_agent
        self.data_dir = Path(data_dir)
        self.log_dir = Path(log_dir)
        self.request_delay = max(request_delay, 0.0)
        self.request_delay_jitter = max(request_delay_jitter, 0.0)
        self.navigation_timeout_ms = navigation_timeout_ms
        self.locale = locale
        self.timezone_id = timezone_id
        self.storage_state = Path(storage_state) if storage_state else None
        self.save_storage_state = Path(save_storage_state) if save_storage_state else None
        self.cookies_file = Path(cookies_file) if cookies_file else None
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        setup_logger(self.log_dir, self.platform, log_level)

        self.playwright: Optional[Playwright] = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self._captured_responses: List[Dict[str, Any]] = []

    def start(self) -> None:
        """启动浏览器。"""
        try:
            from playwright.sync_api import sync_playwright
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "缺少 playwright 依赖，请先执行 `pip install -r requirements.txt`，"
                "然后运行 `python -m playwright install chromium`。"
            ) from exc

        logger.info("启动 {} 浏览器会话", self.platform)
        self.playwright = sync_playwright().start()

        browser_args: Dict[str, Any] = {
            "headless": self.headless,
            "args": [
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-infobars",
            ],
        }
        if self.proxy:
            browser_args["proxy"] = {"server": self.proxy}

        self.browser = self.playwright.chromium.launch(**browser_args)

        context_args: Dict[str, Any] = {
            "locale": self.locale,
            "timezone_id": self.timezone_id,
            "viewport": {
                "width": self.viewport_width,
                "height": self.viewport_height,
            },
            "user_agent": self.user_agent or (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0.0.0 Safari/537.36"
            ),
        }
        if self.storage_state and self.storage_state.exists():
            context_args["storage_state"] = str(self.storage_state)

        self.context = self.browser.new_context(**context_args)
        self._install_context_hooks()
        self._load_cookies_if_needed()

        self.page = self.context.new_page()
        self.page.set_default_timeout(self.navigation_timeout_ms)
        self.page.set_default_navigation_timeout(self.navigation_timeout_ms)
        self.page.add_init_script(
            """
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            Object.defineProperty(navigator, 'platform', { get: () => 'MacIntel' });
            """
        )

        logger.info("{} 浏览器启动完成", self.platform)

    def _install_context_hooks(self) -> None:
        if self.context:
            self.context.on("response", self._capture_response)

    def _capture_response(self, response: Response) -> None:
        if not self.response_patterns:
            return

        url = response.url.lower()
        if not any(pattern in url for pattern in self.response_patterns):
            return

        content_type = response.headers.get("content-type", "").lower()
        if "json" not in content_type and ".json" not in url:
            return

        try:
            payload = response.json()
        except Exception:
            return

        self._captured_responses.append(
            {
                "url": response.url,
                "status": response.status,
                "payload": payload,
            }
        )

    def _load_cookies_if_needed(self) -> None:
        if not self.context or not self.cookies_file or not self.cookies_file.exists():
            return

        with self.cookies_file.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        cookies = payload if isinstance(payload, list) else payload.get("cookies", [])
        if cookies:
            self.context.add_cookies(cookies)
            logger.info("已加载 {} 个 cookies", len(cookies))

    def close(self) -> None:
        """关闭浏览器。"""
        if self.context and self.save_storage_state:
            self.save_storage_state.parent.mkdir(parents=True, exist_ok=True)
            self.context.storage_state(path=str(self.save_storage_state))
            logger.info("已保存 storage state: {}", self.save_storage_state)

        if self.page:
            self.page.close()
        if self.context:
            self.context.close()
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()

        self.page = None
        self.context = None
        self.browser = None
        self.playwright = None

    def reset_captured_responses(self) -> None:
        self._captured_responses = []

    def random_sleep(self, min_sec: Optional[float] = None, max_sec: Optional[float] = None) -> None:
        """随机等待，降低固定节奏。"""
        if min_sec is None:
            min_sec = self.request_delay
        if max_sec is None:
            max_sec = self.request_delay + self.request_delay_jitter
        if max_sec < min_sec:
            max_sec = min_sec
        time.sleep(random.uniform(min_sec, max_sec))

    def goto(self, url: str, wait_until: str = "domcontentloaded") -> None:
        """访问页面并等待基本加载完成。"""
        if not self.page:
            raise RuntimeError("浏览器未启动，请先调用 start() 或使用 with 上下文。")

        self.reset_captured_responses()
        logger.info("访问页面: {}", url)
        self.page.goto(url, wait_until=wait_until)

        try:
            self.page.wait_for_load_state("networkidle", timeout=10000)
        except Exception:
            logger.debug("networkidle 未就绪，继续执行")

        self.random_sleep(1.5, 2.5)

    def close_overlays(self, selectors: Iterable[str]) -> None:
        """尝试关闭遮挡弹窗。"""
        if not self.page:
            return

        for selector in selectors:
            try:
                element = self.page.locator(selector).first
                if element.is_visible(timeout=1000):
                    element.click(timeout=1000)
                    self.random_sleep(0.3, 0.8)
                    logger.debug("关闭弹窗: {}", selector)
                    return
            except Exception:
                continue

    def scroll_until_stable(
        self,
        item_selectors: Sequence[str],
        scroll_times: int = 8,
        container_selectors: Optional[Sequence[str]] = None,
        stable_rounds: int = 2,
    ) -> None:
        """滚动评论区直到稳定。"""
        if not self.page:
            return

        item_expr = ",".join(item_selectors)
        container_expr = ",".join(container_selectors or [])
        unchanged_rounds = 0
        previous_count = -1

        for index in range(scroll_times):
            current_count = self.page.evaluate(
                """
                ({ itemExpr }) => document.querySelectorAll(itemExpr).length
                """,
                {"itemExpr": item_expr},
            )

            if container_expr:
                scrolled = self.page.evaluate(
                    """
                    ({ containerExpr }) => {
                        const target = document.querySelector(containerExpr);
                        if (!target) {
                            window.scrollBy(0, window.innerHeight * 0.85);
                            return false;
                        }
                        target.scrollTop = target.scrollHeight;
                        return true;
                    }
                    """,
                    {"containerExpr": container_expr},
                )
                if not scrolled:
                    self.page.mouse.wheel(0, int(self.viewport_height * 0.85))
            else:
                self.page.mouse.wheel(0, int(self.viewport_height * 0.85))

            self.random_sleep(1.2, 2.0)

            new_count = self.page.evaluate(
                """
                ({ itemExpr }) => document.querySelectorAll(itemExpr).length
                """,
                {"itemExpr": item_expr},
            )
            logger.debug("第 {} 次滚动，评论数 {} -> {}", index + 1, current_count, new_count)

            if new_count <= previous_count:
                unchanged_rounds += 1
            else:
                unchanged_rounds = 0
            previous_count = new_count

            if unchanged_rounds >= stable_rounds:
                break

    def export_records(
        self,
        records: List[Dict[str, Any]],
        output_prefix: Optional[str] = None,
        formats: Sequence[str] = ("json", "jsonl", "csv"),
    ) -> Dict[str, str]:
        """导出标准化结果。"""
        prefix = output_prefix or self.platform
        basename = f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"
        return save_records_bundle(records, self.data_dir, basename, formats=formats)

    def crawl_multiple(self, urls: Sequence[str], **kwargs: Any) -> List[Dict[str, Any]]:
        """批量爬取多个链接。"""
        results: List[Dict[str, Any]] = []

        for index, url in enumerate(urls, start=1):
            logger.info("开始爬取 {}/{}: {}", index, len(urls), url)
            try:
                record = self.crawl(url, **kwargs)
                if record:
                    results.append(record)
            except Exception as exc:
                logger.exception("爬取失败 {}: {}", url, exc)

            if index < len(urls):
                self.random_sleep()

        return results

    def build_record(
        self,
        source_url: str,
        post: Optional[Dict[str, Any]] = None,
        comments: Optional[List[Dict[str, Any]]] = None,
        extraction_source: str = "dom",
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """构建统一结构。"""
        record = {
            "platform": self.platform,
            "source_url": source_url,
            "crawled_at": iso_now(),
            "extraction_source": extraction_source,
            "post": post or {},
            "comments": comments or [],
            "statistics": {
                "comment_count": len(comments or []),
            },
        }
        if extra:
            record.update(extra)
        return record

    def merge_post_info(self, primary: Dict[str, Any], secondary: Dict[str, Any]) -> Dict[str, Any]:
        """优先使用 primary 的非空字段。"""
        merged = dict(secondary)
        merged.update({key: value for key, value in primary.items() if value not in ("", None, [], {})})
        return merged

    @abstractmethod
    def crawl(self, url: str, **kwargs: Any) -> Dict[str, Any]:
        """爬取单个页面。"""

    def __enter__(self) -> "BaseCrawler":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
