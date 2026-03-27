"""
抖音爬虫
爬取抖音视频评论和回复数据
"""
import re
import time
import random
from typing import List, Dict, Any, Optional

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from loguru import logger

from .base import BaseCrawler


class DouyinCrawler(BaseCrawler):
    """抖音爬虫，爬取视频评论数据"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.platform = "douyin"
    
    def crawl(
        self,
        url: str,
        max_comments: int = 100,
        scroll_times: int = 5
    ) -> List[Dict[str, Any]]:
        """
        爬取抖音视频评论
        
        Args:
            url: 抖音视频 URL
            max_comments: 最大评论数
            scroll_times: 滚动次数
            
        Returns:
            评论数据列表
        """
        logger.info(f"开始爬取抖音视频：{url}")
        
        try:
            # 访问视频页面
            self.page.goto(url, wait_until="domcontentloaded")
            self.random_sleep(2, 3)
            
            # 关闭可能的登录弹窗
            self._close_login_popup()
            
            # 滚动页面加载评论
            self._load_comments(scroll_times)
            
            # 提取评论数据
            comments = self._extract_comments(max_comments)
            
            logger.info(f"成功爬取 {len(comments)} 条评论")
            return comments
            
        except PlaywrightTimeoutError as e:
            logger.error(f"爬取超时：{e}")
            return []
        except Exception as e:
            logger.error(f"爬取失败：{e}")
            return []
    
    def _close_login_popup(self):
        """关闭登录弹窗"""
        try:
            # 尝试关闭登录引导弹窗
            close_btn = self.page.query_selector("button[aria-label='关闭'], .close-btn")
            if close_btn:
                close_btn.click()
                self.random_sleep(0.5, 1)
                logger.debug("关闭登录弹窗")
        except Exception:
            pass  # 没有弹窗则忽略
    
    def _load_comments(self, scroll_times: int):
        """滚动页面加载评论"""
        try:
            # 找到评论区域并滚动
            for i in range(scroll_times):
                self.page.evaluate("window.scrollBy(0, 300)")
                self.random_sleep(1, 2)
                logger.debug(f"滚动加载评论 {i + 1}/{scroll_times}")
        except Exception as e:
            logger.warning(f"滚动页面失败：{e}")
    
    def _extract_comments(self, max_comments: int) -> List[Dict[str, Any]]:
        """
        提取评论数据
        
        Returns:
            评论数据列表
        """
        comments = []
        
        try:
            # 使用 JavaScript 提取评论数据
            comment_data = self.page.evaluate("""
                () => {
                    const comments = [];
                    const commentElements = document.querySelectorAll('[data-e2e="comment-item"], .comment-item');
                    
                    commentElements.forEach((el, index) => {
                        try {
                            const content = el.querySelector('.comment-text')?.textContent?.trim() || '';
                            const likes = el.querySelector('.like-count')?.textContent?.trim() || '0';
                            const time = el.querySelector('.comment-time')?.textContent?.trim() || '';
                            const user = el.querySelector('.user-name')?.textContent?.trim() || '';
                            
                            // 提取回复
                            const replies = [];
                            const replyElements = el.querySelectorAll('.reply-item');
                            replyElements.forEach(replyEl => {
                                const replyContent = replyEl.querySelector('.reply-text')?.textContent?.trim() || '';
                                const replyLikes = replyEl.querySelector('.like-count')?.textContent?.trim() || '0';
                                replies.push({
                                    content: replyContent,
                                    likes: parseInt(replyLikes) || 0
                                });
                            });
                            
                            comments.push({
                                content: content,
                                likes: parseInt(likes.replace(/[^0-9]/g, '')) || 0,
                                time: time,
                                user: user,
                                replies: replies,
                                reply_count: replies.length
                            });
                        } catch (e) {
                            console.error('提取评论失败:', e);
                        }
                    });
                    
                    return comments;
                }
            """)
            
            comments = comment_data[:max_comments]
            
        except Exception as e:
            logger.error(f"提取评论失败：{e}")
        
        return comments
    
    def crawl_multiple(
        self,
        urls: List[str],
        output_file: Optional[str] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        批量爬取多个视频
        
        Args:
            urls: 视频 URL 列表
            output_file: 输出文件路径
            **kwargs: 传递给 crawl 的参数
            
        Returns:
            所有评论数据
        """
        all_comments = []
        
        for i, url in enumerate(urls):
            logger.info(f"爬取进度：{i + 1}/{len(urls)}")
            comments = self.crawl(url, **kwargs)
            all_comments.extend(comments)
            
            # 随机延迟，避免被封
            if i < len(urls) - 1:
                delay = random.uniform(3, 6)
                logger.info(f"等待 {delay:.1f} 秒...")
                time.sleep(delay)
        
        if output_file:
            self.save_data(all_comments, output_file, self.platform)
        
        return all_comments
