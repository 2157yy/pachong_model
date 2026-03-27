"""
小红书爬虫。
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

from src.crawler.base import BaseCrawler
from src.utils.common import format_timestamp, parse_number, quote_keyword, safe_text_candidates
from src.utils.logger import logger


class XiaohongshuCrawler(BaseCrawler):
    """小红书笔记评论爬虫。"""

    platform = "xiaohongshu"
    response_patterns = (
        "/comment/page",
        "/feed",
        "/note",
        "/note/get",
        "/api/sns/web",
    )

    COMMENT_ITEM_SELECTORS = (
        '[class*="comment-item"]',
        '[class*="CommentItem"]',
        '[data-testid="comment-item"]',
    )
    COMMENT_CONTAINER_SELECTORS = (
        '[class*="comment-list"]',
        '[class*="CommentList"]',
    )
    POPUP_SELECTORS = (
        'button[aria-label="关闭"]',
        '[class*="close"]',
        '[class*="Close"]',
    )

    def discover_urls(self, keyword: str, max_results: int = 20, scroll_times: int = 8) -> List[str]:
        search_url = f"https://www.xiaohongshu.com/search_result?keyword={quote_keyword(keyword)}&source=web_explore_feed"
        self.goto(search_url)
        self.close_overlays(self.POPUP_SELECTORS)
        self._scroll_for_discovery(scroll_times)
        urls = self._extract_discovery_urls(max_results)
        logger.info("小红书关键词 {} 发现 {} 条作品链接", keyword, len(urls))
        return urls

    def crawl(self, url: str, max_comments: int = 100, scroll_times: int = 8) -> Dict[str, Any]:
        self.goto(url)
        self.close_overlays(self.POPUP_SELECTORS)
        self.scroll_until_stable(
            item_selectors=self.COMMENT_ITEM_SELECTORS,
            container_selectors=self.COMMENT_CONTAINER_SELECTORS,
            scroll_times=scroll_times,
        )

        network_record = self._build_from_network(url, max_comments)
        dom_record = self._build_from_dom(url, max_comments)

        post = self.merge_post_info(network_record.get("post", {}), dom_record.get("post", {}))
        comments = network_record.get("comments") or dom_record.get("comments", [])
        extraction_source = "network" if network_record.get("comments") else "dom"
        if network_record.get("comments") and dom_record.get("comments"):
            extraction_source = "network+dom"

        record = self.build_record(
            source_url=url,
            post=post,
            comments=comments[:max_comments],
            extraction_source=extraction_source,
        )
        record["statistics"]["raw_response_count"] = len(self._captured_responses)
        return record

    def _build_from_network(self, url: str, max_comments: int) -> Dict[str, Any]:
        post: Dict[str, Any] = {}
        comments: List[Dict[str, Any]] = []

        for item in self._captured_responses:
            payload = item.get("payload")
            if not isinstance(payload, dict):
                continue

            detail = self._extract_post_from_payload(payload)
            post = self.merge_post_info(detail, post)
            comments.extend(self._extract_comments_from_payload(payload))

        deduped_comments = self._dedupe_comments(comments)[:max_comments]
        return self.build_record(url, post=post, comments=deduped_comments, extraction_source="network")

    def _extract_post_from_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        data = payload.get("data") if isinstance(payload.get("data"), dict) else payload

        items = data.get("items") if isinstance(data, dict) else None
        if isinstance(items, list) and items:
            note_card = items[0].get("note_card") or items[0]
        else:
            note_card = data.get("note_card") if isinstance(data, dict) else None

        if not isinstance(note_card, dict):
            return {}

        user = note_card.get("user") or note_card.get("user_info") or {}
        interact_info = note_card.get("interact_info") or note_card.get("interaction_info") or {}

        return {
            "post_id": note_card.get("note_id") or note_card.get("id") or "",
            "title": safe_text_candidates(note_card.get("title"), note_card.get("display_title")),
            "content": safe_text_candidates(note_card.get("desc"), note_card.get("content"), note_card.get("title")),
            "author": safe_text_candidates(user.get("nickname"), user.get("user_id")),
            "likes": parse_number(interact_info.get("liked_count") or note_card.get("liked_count")),
            "collects": parse_number(interact_info.get("collected_count") or note_card.get("collected_count")),
            "comments": parse_number(interact_info.get("comment_count") or note_card.get("comment_count")),
            "shares": parse_number(interact_info.get("share_count") or note_card.get("share_count")),
            "views": parse_number(interact_info.get("view_count") or note_card.get("view_count")),
            "publish_time": safe_text_candidates(note_card.get("time"), note_card.get("publish_time")),
        }

    def _extract_comments_from_payload(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
        block = data.get("comments") if isinstance(data, dict) else None
        if not isinstance(block, list):
            return []

        comments: List[Dict[str, Any]] = []
        for item in block:
            if not isinstance(item, dict):
                continue

            user = item.get("user_info") or item.get("user") or {}
            replies = []
            for reply in item.get("sub_comments", []) or item.get("replies", []) or []:
                if not isinstance(reply, dict):
                    continue
                reply_user = reply.get("user_info") or reply.get("user") or {}
                replies.append(
                    {
                        "reply_id": reply.get("id") or reply.get("comment_id") or "",
                        "user": safe_text_candidates(reply_user.get("nickname"), reply_user.get("user_id")),
                        "content": safe_text_candidates(reply.get("content"), reply.get("text")),
                        "likes": parse_number(reply.get("like_count")),
                        "time": safe_text_candidates(reply.get("create_time"), reply.get("time")),
                    }
                )

            comments.append(
                {
                    "comment_id": item.get("id") or item.get("comment_id") or "",
                    "user": safe_text_candidates(user.get("nickname"), user.get("user_id")),
                    "content": safe_text_candidates(item.get("content"), item.get("text")),
                    "likes": parse_number(item.get("like_count")),
                    "time": safe_text_candidates(item.get("create_time"), item.get("time")),
                    "reply_count": parse_number(item.get("sub_comment_count") or len(replies)),
                    "replies": replies,
                }
            )

        return comments

    def _build_from_dom(self, url: str, max_comments: int) -> Dict[str, Any]:
        if not self.page:
            return self.build_record(url, extraction_source="dom")

        payload = self.page.evaluate(
            """
            () => {
                const text = (selectorList, root=document) => {
                    for (const selector of selectorList) {
                        const node = root.querySelector(selector);
                        if (node && node.textContent && node.textContent.trim()) {
                            return node.textContent.trim();
                        }
                    }
                    return '';
                };

                const parseCount = (value) => {
                    if (!value) return 0;
                    const text = String(value).replace(/,/g, '').trim().toLowerCase();
                    const match = text.match(/-?\\d+(\\.\\d+)?/);
                    if (!match) return 0;
                    let num = parseFloat(match[0]);
                    if (text.endsWith('w') || text.endsWith('万')) num *= 10000;
                    if (text.endsWith('k') || text.endsWith('千')) num *= 1000;
                    if (text.endsWith('m')) num *= 1000000;
                    return Math.floor(num);
                };

                const itemSelectors = [
                    '[class*="comment-item"]',
                    '[class*="CommentItem"]',
                    '[data-testid="comment-item"]'
                ];
                const commentNodes = [];
                for (const selector of itemSelectors) {
                    const nodes = Array.from(document.querySelectorAll(selector));
                    if (nodes.length) {
                        commentNodes.push(...nodes);
                        break;
                    }
                }

                const comments = commentNodes.map((node) => {
                    const replyNodes = Array.from(node.querySelectorAll('[class*="reply"], [class*="Reply"]'));
                    return {
                        comment_id: node.getAttribute('data-id') || '',
                        user: text(['[class*="name"]', '[class*="user"]', '[class*="nickname"]'], node),
                        content: text(['[class*="content"]', '[class*="text"]'], node),
                        likes: parseCount(text(['[class*="like"]'], node)),
                        time: text(['time', '[class*="time"]', '[class*="date"]'], node),
                        reply_count: replyNodes.length,
                        replies: replyNodes.map((replyNode) => ({
                            reply_id: replyNode.getAttribute('data-id') || '',
                            user: text(['[class*="name"]', '[class*="user"]', '[class*="nickname"]'], replyNode),
                            content: text(['[class*="content"]', '[class*="text"]'], replyNode),
                            likes: parseCount(text(['[class*="like"]'], replyNode)),
                            time: text(['time', '[class*="time"]', '[class*="date"]'], replyNode)
                        }))
                    };
                });

                return {
                    post: {
                        title: text(['h1', '[class*="title"]']),
                        content: text(['[class*="desc"]', '[class*="content"]', 'article']),
                        author: text(['[class*="author"]', '[class*="nickname"]', '[class*="user"]']),
                        likes: parseCount(text(['[class*="like"]', '[class*="liked"]'])),
                        collects: parseCount(text(['[class*="collect"]', '[class*="favorite"]'])),
                        comments: parseCount(text(['[class*="comment"]'])),
                        shares: parseCount(text(['[class*="share"]']))
                    },
                    comments
                };
            }
            """
        )

        post = payload.get("post", {}) if isinstance(payload, dict) else {}
        comments = payload.get("comments", []) if isinstance(payload, dict) else []
        return self.build_record(url, post=post, comments=self._dedupe_comments(comments)[:max_comments], extraction_source="dom")

    def _dedupe_comments(self, comments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        deduped: List[Dict[str, Any]] = []
        seen = set()

        for comment in comments:
            content = safe_text_candidates(comment.get("content"))
            if not content:
                continue
            key = (
                comment.get("comment_id") or "",
                comment.get("user") or "",
                content,
                comment.get("time") or "",
            )
            if key in seen:
                continue
            seen.add(key)

            replies = []
            reply_seen = set()
            for reply in comment.get("replies", []) or []:
                reply_content = safe_text_candidates(reply.get("content"))
                if not reply_content:
                    continue
                reply_key = (
                    reply.get("reply_id") or "",
                    reply.get("user") or "",
                    reply_content,
                    reply.get("time") or "",
                )
                if reply_key in reply_seen:
                    continue
                reply_seen.add(reply_key)
                replies.append(
                    {
                        "reply_id": reply.get("reply_id", ""),
                        "user": reply.get("user", ""),
                        "content": reply_content,
                        "likes": parse_number(reply.get("likes", 0)),
                        "time": safe_text_candidates(reply.get("time")),
                    }
                )

            deduped.append(
                {
                    "comment_id": comment.get("comment_id", ""),
                    "user": comment.get("user", ""),
                    "content": content,
                    "likes": parse_number(comment.get("likes", 0)),
                    "time": safe_text_candidates(comment.get("time")),
                    "reply_count": parse_number(comment.get("reply_count", len(replies))),
                    "replies": replies,
                }
            )

        logger.info("小红书去重后评论数: {}", len(deduped))
        return deduped

    def _scroll_for_discovery(self, scroll_times: int) -> None:
        if not self.page:
            return
        for _ in range(scroll_times):
            self.page.mouse.wheel(0, 1200)
            self.random_sleep(1.0, 1.8)

    def _extract_discovery_urls(self, max_results: int) -> List[str]:
        if not self.page:
            return []

        raw_links = self.page.evaluate(
            """
            () => Array.from(document.querySelectorAll('a[href]'))
                .map((node) => node.href || node.getAttribute('href') || '')
                .filter(Boolean)
            """
        )

        urls: List[str] = []
        seen = set()
        for href in raw_links:
            normalized = self._normalize_discovery_url(href)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            urls.append(normalized)
            if len(urls) >= max_results:
                break
        return urls

    def _normalize_discovery_url(self, href: str) -> str:
        if not href:
            return ""
        href = href.strip()
        if href.startswith("//"):
            href = f"https:{href}"
        if href.startswith("/"):
            href = f"https://www.xiaohongshu.com{href}"
        if "xiaohongshu.com" not in href:
            return ""

        explore_match = re.search(r"https?://www\.xiaohongshu\.com/explore/([A-Za-z0-9]+)", href)
        if explore_match:
            return f"https://www.xiaohongshu.com/explore/{explore_match.group(1)}"

        dis_match = re.search(r"https?://www\.xiaohongshu\.com/discovery/item/([A-Za-z0-9]+)", href)
        if dis_match:
            return f"https://www.xiaohongshu.com/explore/{dis_match.group(1)}"

        note_match = re.search(r"noteId=([A-Za-z0-9]+)", href)
        if note_match:
            return f"https://www.xiaohongshu.com/explore/{note_match.group(1)}"

        return ""
