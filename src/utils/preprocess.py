"""
数据预处理模块
负责数据清洗、格式转换、数据集划分
"""
import json
import re
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter

import pandas as pd
from loguru import logger


class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self, data_dir: str = "data"):
        self.raw_dir = Path(data_dir) / "raw"
        self.processed_dir = Path(data_dir) / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            "logs/preprocess_{time:YYYY-MM-DD}.log",
            rotation="1 day",
            retention="7 days"
        )
    
    def load_raw_data(self, filepath: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        加载原始数据
        
        Args:
            filepath: 文件路径，如 None 则加载所有 raw 数据
            
        Returns:
            原始数据列表
        """
        if filepath:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        
        # 加载所有 raw 数据
        all_data = []
        for file in self.raw_dir.glob("*.json"):
            logger.info(f"加载文件：{file}")
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                all_data.extend(data)
        
        logger.info(f"共加载 {len(all_data)} 条原始数据")
        return all_data
    
    def clean_text(self, text: str) -> str:
        """
        清洗文本
        
        Args:
            text: 原始文本
            
        Returns:
            清洗后的文本
        """
        if not text:
            return ""
        
        # 移除多余空白
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 移除 URL
        text = re.sub(r'http[s]?://\S+', '', text)
        
        # 移除 @用户
        text = re.sub(r'@\S+', '', text)
        
        # 移除话题标签
        text = re.sub(r'#\S+#', '', text)
        
        # 移除表情符号（可选）
        # text = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF]', '', text)
        
        # 移除特殊字符
        text = re.sub(r'[^\w\s\u4e00-\u9fff，。！？、；：""''（）【】《》…—]', '', text)
        
        return text
    
    def filter_comments(
        self,
        data: List[Dict[str, Any]],
        min_length: int = 5,
        max_length: int = 512,
        min_likes: int = 0
    ) -> List[Dict[str, Any]]:
        """
        过滤评论数据
        
        Args:
            data: 原始数据
            min_length: 最小长度
            max_length: 最大长度
            min_likes: 最小点赞数
            
        Returns:
            过滤后的数据
        """
        filtered = []
        
        for item in data:
            content = item.get('content', '')
            likes = item.get('likes', 0)
            
            # 长度过滤
            if len(content) < min_length or len(content) > max_length:
                continue
            
            # 点赞数过滤
            if likes < min_likes:
                continue
            
            # 清洗文本
            item['content'] = self.clean_text(content)
            item['cleaned'] = True
            
            filtered.append(item)
        
        logger.info(f"过滤后剩余 {len(filtered)}/{len(data)} 条数据")
        return filtered
    
    def extract_dialogue_pairs(
        self,
        data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        提取对话对（评论 - 回复对）
        
        Args:
            data: 评论数据
            
        Returns:
            对话对列表
        """
        dialogue_pairs = []
        
        for comment in data:
            content = comment.get('content', '')
            replies = comment.get('replies', [])
            
            # 评论本身作为第一轮
            if content:
                dialogue_pairs.append({
                    'dialogue': [content],
                    'likes': comment.get('likes', 0),
                    'time': comment.get('time', ''),
                    'platform': comment.get('platform', 'unknown')
                })
            
            # 每个回复作为后续轮次
            for reply in replies:
                reply_content = reply.get('content', '')
                if reply_content and len(reply_content) >= 5:
                    dialogue_pairs.append({
                        'dialogue': [content, reply_content] if content else [reply_content],
                        'likes': reply.get('likes', 0),
                        'platform': comment.get('platform', 'unknown')
                    })
        
        logger.info(f"提取 {len(dialogue_pairs)} 个对话对")
        return dialogue_pairs
    
    def split_dataset(
        self,
        data: List[Dict[str, Any]],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        shuffle: bool = True
    ) -> Tuple[List, List, List]:
        """
        划分数据集
        
        Args:
            data: 数据列表
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            shuffle: 是否打乱
            
        Returns:
            (train_data, val_data, test_data)
        """
        if shuffle:
            random.shuffle(data)
        
        n = len(data)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]
        
        logger.info(f"数据集划分：训练集={len(train_data)}, 验证集={len(val_data)}, 测试集={len(test_data)}")
        return train_data, val_data, test_data
    
    def save_dataset(
        self,
        train_data: List,
        val_data: List,
        test_data: List,
        prefix: str = "dataset"
    ):
        """保存数据集"""
        # 保存为 JSONL 格式（适合大模型训练）
        for name, data in [
            ("train", train_data),
            ("val", val_data),
            ("test", test_data)
        ]:
            filepath = self.processed_dir / f"{prefix}_{name}.jsonl"
            with open(filepath, "w", encoding="utf-8") as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            logger.info(f"保存 {len(data)} 条数据到 {filepath}")
    
    def process_all(
        self,
        input_file: Optional[str] = None,
        output_prefix: str = "dataset",
        min_length: int = 10,
        max_length: int = 512
    ):
        """
        完整的数据处理流程
        
        Args:
            input_file: 输入文件（可选）
            output_prefix: 输出文件前缀
            min_length: 最小长度
            max_length: 最大长度
        """
        # 加载数据
        raw_data = self.load_raw_data(input_file)
        
        # 过滤
        filtered_data = self.filter_comments(
            raw_data,
            min_length=min_length,
            max_length=max_length
        )
        
        # 提取对话对
        dialogue_data = self.extract_dialogue_pairs(filtered_data)
        
        # 划分数据集
        train, val, test = self.split_dataset(dialogue_data)
        
        # 保存
        self.save_dataset(train, val, test, output_prefix)
        
        # 统计信息
        self._print_statistics(train, val, test)
        
        return train, val, test
    
    def _print_statistics(self, train: List, val: List, test: List):
        """打印数据统计信息"""
        all_data = train + val + test
        
        lengths = [len(item.get('dialogue', [''])[0]) for item in all_data if item.get('dialogue')]
        likes = [item.get('likes', 0) for item in all_data]
        
        print("\n" + "=" * 50)
        print("数据统计信息")
        print("=" * 50)
        print(f"总样本数：{len(all_data)}")
        print(f"训练集：{len(train)} ({len(train)/len(all_data)*100:.1f}%)")
        print(f"验证集：{len(val)} ({len(val)/len(all_data)*100:.1f}%)")
        print(f"测试集：{len(test)} ({len(test)/len(all_data)*100:.1f}%)")
        print(f"\n文本长度统计:")
        print(f"  平均：{sum(lengths)/len(lengths):.1f} 字")
        print(f"  最短：{min(lengths)} 字")
        print(f"  最长：{max(lengths)} 字")
        print(f"\n点赞数统计:")
        print(f"  平均：{sum(likes)/len(likes):.1f}")
        print(f"  最高：{max(likes)}")
        print("=" * 50 + "\n")
