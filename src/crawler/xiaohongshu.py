"""
小红书爬虫
爬取小红书笔记评论和回复数据
"""
import time
import random
from typing import List, Dict, Any, Optional

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from loguru import logger

from .base import BaseCrawler


class XiaohongshuCrawler(BaseCrawler):
    """小红书爬虫，爬取笔记评论数据"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.platform = "xiaohongshu"
    
    def crawl(
        self,
        url: str,
        max_comments: int = 100,
        scroll_times: int = 5
    ) -> List[Dict[str, Any]]:
        """
        爬取小红书笔记评论
        
        Args:
            url: 小红书笔记 URL
            max_comments: 最大评论数
            scroll_times: 滚动次数
            
        Returns:
            评论数据列表
        """
        logger.info(f"开始爬取小红书笔记：{url}")
        
        try:
            # 访问笔记页面
            self.page.goto(url, wait_until="domcontentloaded")
            self.random_sleep(2, 3)
            
            # 关闭可能的弹窗
            self._close_popup()
            
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
    
    def _close_popup(self):
        """关闭弹窗"""
        try:
            # 尝试关闭登录引导或其他弹窗
            close_selectors = [
                "button[aria-label='关闭']",
                ".close-icon",
                ".x-icon",
                "[class*='close']"
            ]
            for selector in close_selectors:
                close_btn = self.page.query_selector(selector)
                if close_btn:
                    close_btn.click()
                    self.random_sleep(0.5, 1)
                    logger.debug("关闭弹窗")
                    break
        except Exception:
            pass
    
    def _load_comments(self, scroll_times: int):
        """滚动页面加载评论"""
        try:
            for i in range(scroll_times):
                self.page.evaluate("window.scrollBy(0, 400)")
                self.random_sleep(1.5, 2.5)
                logger.debug(f"滚动加载评论 {i + 1}/{scroll_times}")
        except Exception as e:
            logger.warning(f"滚动页面失败：{e}")
    
    def _extract_comments(self, max_comments: int) -> List[Dict[str, Any]]:
        """提取评论数据"""
        comments = []
        
        try:
            comment_data = self.page.evaluate("""
                () => {
                    const comments = [];
                    // 小红书评论元素选择器
                    const commentElements = document.querySelectorAll('[class*="comment-item"], [class*="CommentItem"]');
                    
                    commentElements.forEach((el) => {
                        try {
                            const content = el.querySelector('[class*="text"], [class*="content"]')?.textContent?.trim() || '';
                            const likes = el.querySelector('[class*="like"]')?.textContent?.trim() || '0';
                            const time = el.querySelector('[class*="time"], [class*="date"]')?.textContent?.trim() || '';
                            const user = el.querySelector('[class*="username"], [class*="nickname"]')?.textContent?.trim() || '';
                            
                            // 提取回复
                            const replies = [];
                            const replyElements = el.querySelectorAll('[class*="reply"], [class*="sub-comment"]');
                            replyElements.forEach(replyEl => {
                                const replyContent = replyEl.querySelector('[class*="text"]')?.textContent?.trim() || '';
                                const replyLikes = replyEl.querySelector('[class*="like"]')?.textContent?.trim() || '0';
                                replies.push({
                                    content: replyContent,
                                    likes: parseInt(replyLikes.replace(/[^0-9]/g, '')) || 0
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
        """批量爬取多个笔记"""
        all_comments = []
        
        for i, url in enumerate(urls):
            logger.info(f"爬取进度：{i + 1}/{len(urls)}")
            comments = self.crawl(url, **kwargs)
            all_comments.extend(comments)
            
            if i < len(urls) - 1:
                delay = random.uniform(3, 6)
                logger.info(f"等待 {delay:.1f} 秒...")
                time.sleep(delay)
        
        if output_file:
            self.save_data(all_comments, output_file, self.platform)
        
        return all_comments
