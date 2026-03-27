"""
文本质量分类训练器。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.model.classifier import QualityClassifier
from src.utils.logger import logger


class Trainer:
    """支持 Paddle/sklearn 的统一训练接口。"""

    def __init__(
        self,
        model_name: str = "ernie-3.0-base",
        num_labels: int = 3,
        max_length: int = 512,
        learning_rate: float = 2e-5,
        batch_size: int = 32,
        epochs: int = 5,
        warmup_ratio: float = 0.1,
        device: str = "cpu",
        output_dir: str = "checkpoints",
        backend: str = "auto",
    ):
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.warmup_ratio = warmup_ratio
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.backend = self._resolve_backend(backend)

    def _resolve_backend(self, backend: str) -> str:
        if backend != "auto":
            return backend
        try:
            import paddlenlp  # noqa: F401
            import paddle  # noqa: F401
            return "paddle"
        except Exception:
            return "sklearn"

    def load_jsonl(self, filepath: str | Path) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        with Path(filepath).open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def train(
        self,
        train_data: Sequence[Dict[str, Any]],
        val_data: Sequence[Dict[str, Any]],
        save_name: str = "best_model",
        test_data: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        if self.backend == "sklearn":
            return self._train_sklearn(train_data, val_data, save_name, test_data=test_data)
        if self.backend == "paddle":
            return self._train_paddle(train_data, val_data, save_name, test_data=test_data)
        raise RuntimeError(f"不支持的训练后端: {self.backend}")

    def _prepare_xy(self, data: Sequence[Dict[str, Any]]) -> tuple[List[str], List[int]]:
        texts: List[str] = []
        labels: List[int] = []
        for item in data:
            text = item.get("text") or " [SEP] ".join(item.get("dialogue", [])) or item.get("content", "")
            if not text:
                continue
            label = item.get("label_id", item.get("label", 0))
            if isinstance(label, str):
                label = QualityClassifier.LABEL_MAP_REVERSE.get(label, 0)
            texts.append(text)
            labels.append(int(label))
        return texts, labels

    def _metrics(self, y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        return {
            "accuracy": float(accuracy_score(y_true, y_pred)) if y_true else 0.0,
            "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)) if y_true else 0.0,
            "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)) if y_true else 0.0,
            "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if y_true else 0.0,
        }

    def _train_sklearn(
        self,
        train_data: Sequence[Dict[str, Any]],
        val_data: Sequence[Dict[str, Any]],
        save_name: str,
        test_data: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.pipeline import Pipeline

        x_train, y_train = self._prepare_xy(train_data)
        x_val, y_val = self._prepare_xy(val_data)
        x_test, y_test = self._prepare_xy(test_data or [])

        pipeline = Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=50000)),
                (
                    "clf",
                    MultinomialNB(),
                ),
            ]
        )
        pipeline.fit(x_train, y_train)

        val_pred = pipeline.predict(x_val) if x_val else []
        val_metrics = self._metrics(y_val, val_pred)
        test_metrics = self._metrics(y_test, pipeline.predict(x_test)) if x_test else {}

        model_path = self.output_dir / save_name
        classifier = QualityClassifier(
            model_name=self.model_name,
            num_labels=self.num_labels,
            max_length=self.max_length,
            device=self.device,
            backend="sklearn",
        )
        classifier.pipeline = pipeline
        classifier.save(str(model_path))

        history = {
            "backend": "sklearn",
            "train_size": len(x_train),
            "val_size": len(x_val),
            "test_size": len(x_test),
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        }
        with (model_path / "training_history.json").open("w", encoding="utf-8") as file:
            json.dump(history, file, ensure_ascii=False, indent=2)

        logger.info("sklearn 训练完成，模型目录: {}", model_path)
        return history

    def _train_paddle(
        self,
        train_data: Sequence[Dict[str, Any]],
        val_data: Sequence[Dict[str, Any]],
        save_name: str,
        test_data: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        import numpy as np
        import paddle
        from paddle.io import DataLoader, Dataset
        from paddlenlp.data import Dict as PaddleDict
        from paddlenlp.data import Pad, Stack
        from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer

        try:
            paddle.set_device(self.device)
        except Exception:
            paddle.set_device("cpu")
            self.device = "cpu"

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_classes=self.num_labels)

        class SimpleDataset(Dataset):
            def __init__(self, data: Sequence[Dict[str, Any]]):
                self.data = list(data)

            def __len__(self) -> int:
                return len(self.data)

            def __getitem__(self, idx: int) -> Dict[str, Any]:
                item = self.data[idx]
                text = item.get("text") or " [SEP] ".join(item.get("dialogue", [])) or item.get("content", "")
                label = item.get("label_id", item.get("label", 0))
                if isinstance(label, str):
                    label = QualityClassifier.LABEL_MAP_REVERSE.get(label, 0)
                encoded = tokenizer(text=text, max_length=self.max_length, truncation=True)
                encoded.setdefault("token_type_ids", [0] * len(encoded["input_ids"]))
                encoded["labels"] = int(label)
                return encoded

        batchify_fn = PaddleDict(
            {
                "input_ids": Pad(axis=0, pad_val=tokenizer.pad_token_id),
                "token_type_ids": Pad(axis=0, pad_val=getattr(tokenizer, "pad_token_type_id", 0) or 0),
                "labels": Stack(dtype="int64"),
            }
        )

        train_loader = DataLoader(
            SimpleDataset(train_data),
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=batchify_fn,
        )
        val_loader = DataLoader(
            SimpleDataset(val_data),
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=batchify_fn,
        )

        loss_fn = paddle.nn.CrossEntropyLoss()
        optimizer = paddle.optimizer.AdamW(
            learning_rate=self.learning_rate,
            parameters=model.parameters(),
        )

        best_f1 = -1.0
        best_path = self.output_dir / save_name

        for epoch in range(self.epochs):
            model.train()
            for batch in train_loader:
                logits = model(
                    input_ids=batch["input_ids"],
                    token_type_ids=batch["token_type_ids"],
                )[0]
                loss = loss_fn(logits, batch["labels"])
                loss.backward()
                optimizer.step()
                optimizer.clear_grad()

            metrics = self._evaluate_paddle(model, val_loader)
            logger.info("Epoch {}/{} val_f1={:.4f}", epoch + 1, self.epochs, metrics["f1"])
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_path.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(str(best_path))
                tokenizer.save_pretrained(str(best_path))
                with (best_path / "config.json").open("w", encoding="utf-8") as file:
                    json.dump(
                        {
                            "backend": "paddle",
                            "model_name": self.model_name,
                            "num_labels": self.num_labels,
                            "max_length": self.max_length,
                            "device": self.device,
                        },
                        file,
                        ensure_ascii=False,
                        indent=2,
                    )

        history = {
            "backend": "paddle",
            "best_val_f1": best_f1,
            "train_size": len(train_data),
            "val_size": len(val_data),
            "test_size": len(test_data or []),
        }
        with (best_path / "training_history.json").open("w", encoding="utf-8") as file:
            json.dump(history, file, ensure_ascii=False, indent=2)

        if test_data:
            test_loader = DataLoader(
                SimpleDataset(test_data),
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=batchify_fn,
            )
            history["test_metrics"] = self._evaluate_paddle(model, test_loader)

        logger.info("paddle 训练完成，模型目录: {}", best_path)
        return history

    def _evaluate_paddle(self, model: Any, data_loader: Any) -> Dict[str, float]:
        import numpy as np
        import paddle
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        model.eval()
        y_true: List[int] = []
        y_pred: List[int] = []
        with paddle.no_grad():
            for batch in data_loader:
                logits = model(
                    input_ids=batch["input_ids"],
                    token_type_ids=batch["token_type_ids"],
                )[0]
                preds = np.argmax(logits.numpy(), axis=-1).tolist()
                labels = batch["labels"].numpy().tolist()
                y_true.extend(labels)
                y_pred.extend(preds)

        return {
            "accuracy": float(accuracy_score(y_true, y_pred)) if y_true else 0.0,
            "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)) if y_true else 0.0,
            "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)) if y_true else 0.0,
            "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if y_true else 0.0,
        }
