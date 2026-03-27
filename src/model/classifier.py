"""
质量分类模型
使用 PaddleNLP 加载预训练模型进行文本分类
"""
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import paddle
from paddlenlp.transformers import ErnieTokenizer, ErnieForSequenceClassification
from loguru import logger


class QualityClassifier:
    """对话质量分类器"""
    
    LABEL_MAP = {0: 'low', 1: 'medium', 2: 'high'}
    LABEL_MAP_REVERSE = {'low': 0, 'medium': 1, 'high': 2}
    
    def __init__(
        self,
        model_name: str = "ernie-3.0-base",
        num_labels: int = 3,
        max_length: int = 512,
        device: str = "iluvatar_gpu:0"
    ):
        """
        初始化分类器
        
        Args:
            model_name: 预训练模型名称
            num_labels: 分类数量
            max_length: 最大序列长度
            device: 设备类型
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.device = device
        
        # 设置设备
        try:
            paddle.set_device(device)
            logger.info(f"使用设备：{device}")
        except ValueError:
            # 如果 Iluvatar 不可用，回退到 CPU
            paddle.set_device("cpu")
            self.device = "cpu"
            logger.warning(f"设备 {device} 不可用，使用 CPU")
        
        # 加载模型和分词器
        logger.info(f"加载模型：{model_name}")
        self.tokenizer = ErnieTokenizer.from_pretrained(model_name)
        self.model = ErnieForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        
        logger.info("模型加载完成")
    
    def predict(self, text: str) -> Tuple[str, float]:
        """
        预测单个样本
        
        Args:
            text: 输入文本
            
        Returns:
            (标签，置信度)
        """
        # 分词
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # 预测
        self.model.eval()
        with paddle.no_grad():
            outputs = self.model(**inputs)
            logits = outputs[0]
            probs = paddle.nn.functional.softmax(logits, axis=-1)
        
        # 获取结果
        pred_id = paddle.argmax(probs, axis=-1).item()
        confidence = probs[0][pred_id].item()
        
        label = self.LABEL_MAP.get(pred_id, 'unknown')
        
        return label, confidence
    
    def predict_batch(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> List[Tuple[str, float]]:
        """
        批量预测
        
        Args:
            texts: 文本列表
            batch_size: 批次大小
            
        Returns:
            [(标签，置信度), ...]
        """
        results = []
        
        self.model.eval()
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # 分词
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # 预测
            with paddle.no_grad():
                outputs = self.model(**inputs)
                logits = outputs[0]
                probs = paddle.nn.functional.softmax(logits, axis=-1)
            
            # 获取结果
            pred_ids = paddle.argmax(probs, axis=-1).tolist()
            confidences = paddle.max(probs, axis=-1)[0].tolist()
            
            for pred_id, conf in zip(pred_ids, confidences):
                label = self.LABEL_MAP.get(pred_id, 'unknown')
                results.append((label, conf))
        
        return results
    
    def save(self, save_dir: str):
        """保存模型"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(str(save_path))
        self.tokenizer.save_pretrained(str(save_path))
        
        logger.info(f"模型已保存到 {save_path}")
    
    def load(self, model_dir: str):
        """加载模型"""
        model_path = Path(model_dir)
        
        self.model = ErnieForSequenceClassification.from_pretrained(str(model_path))
        self.tokenizer = ErnieTokenizer.from_pretrained(str(model_path))
        
        logger.info(f"模型已从 {model_path} 加载")
