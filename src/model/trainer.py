"""
模型训练器
负责模型训练、评估、保存
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import paddle
from paddle.io import Dataset, DataLoader
from paddlenlp.transformers import ErnieTokenizer, ErnieForSequenceClassification
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.metrics import Accuracy, Precision, Recall, F1
from loguru import logger


class TextClassificationDataset(Dataset):
    """文本分类数据集"""
    
    def __init__(self, data: List[Dict[str, Any]], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item.get('text', '') or item.get('dialogue', [''])[0] or ''
        label = item.get('label_id', item.get('label', 0))
        
        # 标签映射
        if isinstance(label, str):
            label_map = {'low': 0, 'medium': 1, 'high': 2}
            label = label_map.get(label, 0)
        
        # 分词
        encoded = self.tokenizer(
            text,
            padding=False,
            truncation=True,
            max_length=self.max_length
        )
        
        return {
            'input_ids': encoded['input_ids'],
            'token_type_ids': encoded['token_type_ids'],
            'labels': label
        }


class Trainer:
    """模型训练器"""
    
    def __init__(
        self,
        model_name: str = "ernie-3.0-base",
        num_labels: int = 3,
        max_length: int = 512,
        learning_rate: float = 2e-5,
        batch_size: int = 32,
        epochs: int = 5,
        warmup_ratio: float = 0.1,
        device: str = "iluvatar_gpu:0",
        output_dir: str = "checkpoints"
    ):
        """
        初始化训练器
        
        Args:
            model_name: 预训练模型名称
            num_labels: 分类数量
            max_length: 最大序列长度
            learning_rate: 学习率
            batch_size: 批次大小
            epochs: 训练轮数
            warmup_ratio: 预热比例
            device: 设备类型
            output_dir: 输出目录
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.warmup_ratio = warmup_ratio
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置设备
        try:
            paddle.set_device(device)
            self.device = device
            logger.info(f"使用设备：{device}")
        except ValueError:
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
        
        # 优化器
        self.optimizer = paddle.optimizer.AdamW(
            learning_rate=learning_rate,
            parameters=self.model.parameters()
        )
        
        # 损失函数
        self.criterion = paddle.nn.loss.CrossEntropyLoss()
        
        logger.info("训练器初始化完成")
    
    def create_dataloader(
        self,
        data: List[Dict[str, Any]],
        shuffle: bool = False
    ) -> DataLoader:
        """创建数据加载器"""
        dataset = TextClassificationDataset(data, self.tokenizer, self.max_length)
        
        batch_sampler = paddle.io.DistributedBatchSampler(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle
        )
        
        dataloader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=self._collate_fn
        )
        
        return dataloader
    
    def _collate_fn(self, batch):
        """批次数据处理"""
        input_ids = Stack()(x['input_ids'] for x in batch)
        token_type_ids = Stack()(x['token_type_ids'] for x in batch)
        labels = Stack()(x['labels'] for x in batch)
        
        return input_ids, token_type_ids, labels
    
    def train(
        self,
        train_data: List[Dict[str, Any]],
        val_data: List[Dict[str, Any]],
        save_name: str = "best_model"
    ) -> Dict[str, Any]:
        """
        训练模型
        
        Args:
            train_data: 训练数据
            val_data: 验证数据
            save_name: 保存模型名称
            
        Returns:
            训练历史记录
        """
        # 创建数据加载器
        train_loader = self.create_dataloader(train_data, shuffle=True)
        val_loader = self.create_dataloader(val_data, shuffle=False)
        
        # 计算步数
        num_train_steps = len(train_loader) * self.epochs
        num_warmup_steps = int(num_train_steps * self.warmup_ratio)
        
        # 学习率调度器
        lr_scheduler = paddle.optimizer.lr.LinearWarmup(
            learning_rate=self.learning_rate,
            warmup_steps=num_warmup_steps,
            start_lr=0.0,
            end_lr=self.learning_rate,
            total_steps=num_train_steps
        )
        self.optimizer.set_lr_scheduler(lr_scheduler)
        
        # 训练指标
        best_f1 = 0.0
        history = {'train_loss': [], 'val_f1': [], 'val_accuracy': []}
        
        logger.info(f"开始训练，共 {self.epochs} 轮，{num_train_steps} 步")
        
        for epoch in range(self.epochs):
            # 训练
            self.model.train()
            train_loss_sum = 0.0
            
            for step, (input_ids, token_type_ids, labels) in enumerate(train_loader):
                # 前向传播
                outputs = self.model(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids
                )
                logits = outputs[0]
                
                # 计算损失
                loss = self.criterion(logits, labels)
                
                # 反向传播
                loss.backward()
                self.optimizer.step()
                self.optimizer.clear_grad()
                
                train_loss_sum += loss.item()
                
                # 打印进度
                if (step + 1) % 50 == 0:
                    avg_loss = train_loss_sum / (step + 1)
                    logger.info(f"Epoch {epoch + 1}/{self.epochs}, Step {step + 1}, Loss: {avg_loss:.4f}")
            
            # 平均训练损失
            avg_train_loss = train_loss_sum / len(train_loader)
            history['train_loss'].append(avg_train_loss)
            
            # 验证
            val_metrics = self.evaluate(val_loader)
            history['val_f1'].append(val_metrics['f1'])
            history['val_accuracy'].append(val_metrics['accuracy'])
            
            logger.info(
                f"Epoch {epoch + 1}/{self.epochs} - "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Val Accuracy: {val_metrics['accuracy']:.4f}, "
                f"Val F1: {val_metrics['f1']:.4f}"
            )
            
            # 保存最佳模型
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                self.save(f"{self.output_dir}/{save_name}")
                logger.info(f"保存最佳模型 (F1={best_f1:.4f})")
        
        # 保存训练历史
        with open(f"{self.output_dir}/training_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        
        logger.info("训练完成")
        return history
    
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """
        评估模型
        
        Returns:
            评估指标
        """
        self.model.eval()
        
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with paddle.no_grad():
            for input_ids, token_type_ids, labels in data_loader:
                outputs = self.model(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids
                )
                logits = outputs[0]
                
                # 预测
                preds = paddle.argmax(logits, axis=-1)
                
                correct += (preds == labels).sum().item()
                total += labels.shape[0]
                
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.tolist())
        
        # 计算指标
        accuracy = correct / total if total > 0 else 0
        
        # F1 分数（宏平均）
        from sklearn.metrics import f1_score, precision_score, recall_score
        
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def save(self, save_dir: str):
        """保存模型"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(str(save_path))
        self.tokenizer.save_pretrained(str(save_path))
        
        # 保存配置
        config = {
            'model_name': self.model_name,
            'num_labels': self.num_labels,
            'max_length': self.max_length,
            'device': self.device
        }
        with open(f"{save_path}/config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"模型已保存到 {save_path}")
