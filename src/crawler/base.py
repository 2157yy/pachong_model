"""
爬虫基础类
定义爬虫的通用接口和方法
"""
import json
import time
import random
from pathlib import Path
from datetime import datetime
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from playwright.sync_api import sync_playwright, Page, Browser, BrowserContext
from loguru import logger


class BaseCrawler(ABC):
    """爬虫基类，提供通用的浏览器操作和数据保存功能"""
    
    def __init__(
        self,
        headless: bool = False,
        proxy: Optional[str] = None,
        user_agent: Optional[str] = None,
        data_dir: str = "data/raw",
    ):
        """
        初始化爬虫
        
        Args:
            headless: 是否无头模式
            proxy: 代理地址
            user_agent: 用户代理
            data_dir: 数据保存目录
        """
        self.headless = headless
        self.proxy = proxy
        self.user_agent = user_agent
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        
        logger.add(
            "logs/crawler_{time:YYYY-MM-DD}.log",
            rotation="1 day",
            retention="7 days",
            level="DEBUG"
        )
    
    def start(self):
        """启动浏览器"""
        logger.info("启动浏览器...")
        self.playwright = sync_playwright().start()
        
        # 浏览器配置
        browser_args = {
            "headless": self.headless,
            "args": [
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--disable-accelerated-2d-canvas",
                "--disable-gpu",
            ]
        }
        
        if self.proxy:
            browser_args["proxy"] = {"server": self.proxy}
        
        self.browser = self.playwright.chromium.launch(**browser_args)
        
        # 上下文配置
        context_args = {}
        if self.user_agent:
            context_args["user_agent"] = self.user_agent
        else:
            # 默认使用桌面端 User-Agent
            context_args["user_agent"] = (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        
        self.context = self.browser.new_context(**context_args)
        self.page = self.context.new_page()
        
        # 设置超时
        self.page.set_default_timeout(30000)
        self.page.set_default_navigation_timeout(30000)
        
        logger.info("浏览器启动完成")
    
    def close(self):
        """关闭浏览器"""
        logger.info("关闭浏览器...")
        if self.page:
            self.page.close()
        if self.context:
            self.context.close()
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()
        logger.info("浏览器已关闭")
    
    def random_sleep(self, min_sec: float = 1.0, max_sec: float = 3.0):
        """随机延迟，模拟人类行为"""
        delay = random.uniform(min_sec, max_sec)
        time.sleep(delay)
    
    def scroll_page(self, times: int = 3, sleep_sec: float = 1.0):
        """滚动页面，加载动态内容"""
        for i in range(times):
            self.page.evaluate("window.scrollBy(0, document.body.scrollHeight)")
            self.random_sleep(sleep_sec, sleep_sec + 0.5)
            logger.debug(f"滚动页面 {i + 1}/{times}")
    
    def save_data(
        self,
        data: List[Dict[str, Any]],
        filename: Optional[str] = None,
        platform: str = "unknown"
    ) -> str:
        """
        保存数据到 JSON 文件
        
        Args:
            data: 数据列表
            filename: 文件名（可选）
            platform: 平台名称
            
        Returns:
            保存的文件路径
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{platform}_{timestamp}.json"
        
        filepath = self.data_dir / filename
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"保存 {len(data)} 条数据到 {filepath}")
        return str(filepath)
    
    def load_data(self, filepath: str) -> List[Dict[str, Any]]:
        """从 JSON 文件加载数据"""
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    
    @abstractmethod
    def crawl(self, url: str, **kwargs) -> List[Dict[str, Any]]:
        """
        爬取数据（子类必须实现）
        
        Args:
            url: 目标 URL
            **kwargs: 其他参数
            
        Returns:
            爬取的数据列表
        """
        pass
    
    def __enter__(self):
        """上下文管理器入口"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()
