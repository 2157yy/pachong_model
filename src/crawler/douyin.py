"""
抖音爬虫。
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

from src.crawler.base import BaseCrawler
from src.utils.common import format_timestamp, parse_number, quote_keyword, safe_text_candidates
from src.utils.logger import logger


class DouyinCrawler(BaseCrawler):
    """抖音评论爬虫。"""

    platform = "douyin"
    response_patterns = (
        "comment/list",
        "comment/list/reply",
        "aweme/detail",
        "web/aweme/detail",
        "web/comment",
    )

    COMMENT_ITEM_SELECTORS = (
        '[data-e2e="comment-item"]',
        '[class*="comment-item"]',
        '[class*="CommentItem"]',
    )
    COMMENT_CONTAINER_SELECTORS = (
        '[data-e2e="comment-list"]',
        '[class*="comment-list"]',
        '[class*="CommentList"]',
    )
    POPUP_SELECTORS = (
        'button[aria-label="关闭"]',
        '[class*="close"]',
        '[class*="Close"]',
    )

    def discover_urls(self, keyword: str, max_results: int = 20, scroll_times: int = 8) -> List[str]:
        search_url = f"https://www.douyin.com/search/{quote_keyword(keyword)}?type=video"
        self.goto(search_url)
        self.close_overlays(self.POPUP_SELECTORS)
        self._scroll_for_discovery(scroll_times)
        urls = self._extract_discovery_urls(max_results)
        logger.info("抖音关键词 {} 发现 {} 条作品链接", keyword, len(urls))
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
        detail = (
            payload.get("aweme_detail")
            or payload.get("data", {}).get("aweme_detail")
            or payload.get("detail")
        )
        if not detail and isinstance(payload.get("aweme_list"), list) and payload["aweme_list"]:
            detail = payload["aweme_list"][0]
        if not isinstance(detail, dict):
            return {}

        statistics = detail.get("statistics") or detail.get("aweme_statistics") or {}
        author = detail.get("author") or detail.get("user") or {}

        return {
            "post_id": detail.get("aweme_id") or detail.get("group_id") or detail.get("item_id") or "",
            "title": safe_text_candidates(detail.get("desc"), detail.get("preview_title")),
            "content": safe_text_candidates(detail.get("desc"), detail.get("preview_title")),
            "author": safe_text_candidates(author.get("nickname"), author.get("unique_id")),
            "likes": parse_number(statistics.get("digg_count") or detail.get("digg_count")),
            "collects": parse_number(statistics.get("collect_count") or detail.get("collect_count")),
            "comments": parse_number(statistics.get("comment_count") or detail.get("comment_count")),
            "shares": parse_number(statistics.get("share_count") or detail.get("share_count")),
            "views": parse_number(statistics.get("play_count") or statistics.get("view_count") or detail.get("play_count")),
            "publish_time": format_timestamp(detail.get("create_time")),
        }

    def _extract_comments_from_payload(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        block = payload.get("comments")
        if not block and isinstance(payload.get("data"), dict):
            block = payload["data"].get("comments")
        if not isinstance(block, list):
            return []

        comments: List[Dict[str, Any]] = []
        for item in block:
            if not isinstance(item, dict):
                continue

            replies = []
            for reply in item.get("reply_comment", []) or item.get("reply_comment_list", []) or []:
                if not isinstance(reply, dict):
                    continue
                reply_user = reply.get("user") or {}
                replies.append(
                    {
                        "reply_id": reply.get("cid") or reply.get("reply_id") or "",
                        "user": safe_text_candidates(reply_user.get("nickname"), reply_user.get("unique_id")),
                        "content": safe_text_candidates(reply.get("text"), reply.get("content")),
                        "likes": parse_number(reply.get("digg_count") or reply.get("like_count")),
                        "time": format_timestamp(reply.get("create_time")),
                    }
                )

            user = item.get("user") or {}
            comments.append(
                {
                    "comment_id": item.get("cid") or item.get("comment_id") or "",
                    "user": safe_text_candidates(user.get("nickname"), user.get("unique_id")),
                    "content": safe_text_candidates(item.get("text"), item.get("content")),
                    "likes": parse_number(item.get("digg_count") or item.get("like_count")),
                    "time": format_timestamp(item.get("create_time")),
                    "reply_count": parse_number(item.get("reply_comment_total") or len(replies)),
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
                    '[data-e2e="comment-item"]',
                    '[class*="comment-item"]',
                    '[class*="CommentItem"]'
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
                        user: text(['[data-e2e="comment-user-name"]', '[class*="name"]', '[class*="user"]'], node),
                        content: text(['[data-e2e="comment-item-text"]', '[class*="text"]', '[class*="content"]'], node),
                        likes: parseCount(text(['[data-e2e="comment-like-count"]', '[class*="like"]'], node)),
                        time: text(['time', '[class*="time"]', '[class*="date"]'], node),
                        reply_count: replyNodes.length,
                        replies: replyNodes.map((replyNode) => ({
                            reply_id: replyNode.getAttribute('data-id') || '',
                            user: text(['[class*="name"]', '[class*="user"]'], replyNode),
                            content: text(['[class*="text"]', '[class*="content"]'], replyNode),
                            likes: parseCount(text(['[class*="like"]'], replyNode)),
                            time: text(['time', '[class*="time"]', '[class*="date"]'], replyNode)
                        }))
                    };
                });

                return {
                    post: {
                        title: document.title || '',
                        content: text(['h1', '[data-e2e="video-desc"]', '[class*="desc"]', '[class*="title"]']),
                        author: text(['[data-e2e="video-author-name"]', '[class*="author"]', '[class*="name"]']),
                        likes: parseCount(text(['[data-e2e="like-count"]', '[class*="digg"]', '[class*="like"]'])),
                        comments: parseCount(text(['[data-e2e="comment-count"]', '[class*="comment"]'])),
                        shares: parseCount(text(['[data-e2e="share-count"]', '[class*="share"]']))
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
                        "time": reply.get("time", ""),
                    }
                )

            deduped.append(
                {
                    "comment_id": comment.get("comment_id", ""),
                    "user": comment.get("user", ""),
                    "content": content,
                    "likes": parse_number(comment.get("likes", 0)),
                    "time": comment.get("time", ""),
                    "reply_count": parse_number(comment.get("reply_count", len(replies))),
                    "replies": replies,
                }
            )

        logger.info("抖音去重后评论数: {}", len(deduped))
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
            href = f"https://www.douyin.com{href}"
        if "douyin.com" not in href:
            return ""

        video_match = re.search(r"https?://www\.douyin\.com/video/(\d+)", href)
        if video_match:
            return f"https://www.douyin.com/video/{video_match.group(1)}"

        modal_match = re.search(r"modal_id=(\d+)", href)
        if modal_match:
            return f"https://www.douyin.com/video/{modal_match.group(1)}"

        note_match = re.search(r"https?://www\.douyin\.com/note/(\d+)", href)
        if note_match:
            return f"https://www.douyin.com/note/{note_match.group(1)}"

        return ""
