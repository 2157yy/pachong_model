from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.crawler.douyin import DouyinCrawler
from src.crawler.xiaohongshu import XiaohongshuCrawler
from src.model.trainer import Trainer
from src.utils.common import flatten_page_records, parse_number
from src.utils.labeler import PseudoLabeler
from src.utils.preprocess import DataPreprocessor


SAMPLE_RECORD = {
    "platform": "douyin",
    "source_url": "https://www.douyin.com/video/demo",
    "crawled_at": "2026-03-27T10:00:00",
    "post": {
        "post_id": "p1",
        "title": "测试标题",
        "content": "测试正文",
        "author": "作者A",
        "likes": 1200,
        "collects": 80,
        "comments": 45,
        "views": 20000,
    },
    "comments": [
        {
            "comment_id": "c1",
            "user": "用户1",
            "content": "这条评论内容足够长，适合作为样本。",
            "likes": 66,
            "time": "2026-03-27T10:00:00",
            "reply_count": 1,
            "replies": [
                {
                    "reply_id": "r1",
                    "user": "用户2",
                    "content": "这条回复也足够长，可以组成对话样本。",
                    "likes": 23,
                    "time": "2026-03-27T10:05:00",
                }
            ],
        }
    ],
}


class PipelineTests(unittest.TestCase):
    def test_parse_number(self) -> None:
        self.assertEqual(parse_number("1.2w"), 12000)
        self.assertEqual(parse_number("3万"), 30000)
        self.assertEqual(parse_number("56"), 56)

    def test_flatten_and_label(self) -> None:
        flat = flatten_page_records([SAMPLE_RECORD])
        self.assertEqual(len(flat), 2)
        labeler = PseudoLabeler()
        labeled = labeler.generate_labels(flat)
        self.assertTrue(all("label" in item for item in labeled))

    def test_preprocess(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            raw_dir = data_dir / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            (raw_dir / "sample.json").write_text("[{}]".format(__import__("json").dumps(SAMPLE_RECORD, ensure_ascii=False)), encoding="utf-8")

            preprocessor = DataPreprocessor(data_dir=str(data_dir), processed_dir=str(data_dir / "processed"))
            result = preprocessor.process_all(output_prefix="demo", labeler=PseudoLabeler())
            self.assertGreater(len(result["train"]) + len(result["val"]) + len(result["test"]), 0)

    def test_douyin_network_parse(self) -> None:
        crawler = DouyinCrawler()
        payload = {
            "aweme_detail": {
                "aweme_id": "1",
                "desc": "作品说明",
                "statistics": {"digg_count": 10, "comment_count": 2},
                "author": {"nickname": "作者"},
                "create_time": 1700000000,
            },
            "comments": [
                {
                    "cid": "c1",
                    "text": "评论内容",
                    "digg_count": 3,
                    "create_time": 1700000001,
                    "user": {"nickname": "用户"},
                }
            ],
        }
        self.assertEqual(crawler._extract_post_from_payload(payload)["post_id"], "1")
        self.assertEqual(len(crawler._extract_comments_from_payload(payload)), 1)

    def test_xiaohongshu_network_parse(self) -> None:
        crawler = XiaohongshuCrawler()
        payload = {
            "data": {
                "items": [
                    {
                        "note_card": {
                            "note_id": "n1",
                            "title": "标题",
                            "desc": "正文",
                            "user": {"nickname": "作者"},
                            "interact_info": {"liked_count": "12", "comment_count": "3"},
                        }
                    }
                ],
                "comments": [
                    {
                        "id": "c1",
                        "content": "评论内容",
                        "like_count": "8",
                        "user_info": {"nickname": "用户"},
                    }
                ],
            }
        }
        self.assertEqual(crawler._extract_post_from_payload(payload)["post_id"], "n1")
        self.assertEqual(len(crawler._extract_comments_from_payload(payload)), 1)

    def test_sklearn_trainer(self) -> None:
        train_data = [
            {"text": "互动很高 内容很好", "label_id": 2},
            {"text": "一般般的评论", "label_id": 1},
            {"text": "广告水评", "label_id": 0},
            {"text": "优质长回复有讨论", "label_id": 2},
            {"text": "普通问答", "label_id": 1},
            {"text": "垃圾内容", "label_id": 0},
        ]
        val_data = [
            {"text": "很优质的讨论", "label_id": 2},
            {"text": "普通对话", "label_id": 1},
            {"text": "低质广告", "label_id": 0},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(output_dir=tmpdir, backend="sklearn")
            history = trainer.train(train_data, val_data, save_name="demo")
            self.assertEqual(history["backend"], "sklearn")
            self.assertTrue((Path(tmpdir) / "demo" / "model.joblib").exists())


if __name__ == "__main__":
    unittest.main()
