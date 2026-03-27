"""
质量分类模型推理封装。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logger import logger


class QualityClassifier:
    """统一封装 Paddle 和 sklearn 两种推理后端。"""

    LABEL_MAP = {0: "low", 1: "medium", 2: "high"}
    LABEL_MAP_REVERSE = {"low": 0, "medium": 1, "high": 2}

    def __init__(
        self,
        model_dir: Optional[str] = None,
        model_name: str = "ernie-3.0-base",
        num_labels: int = 3,
        max_length: int = 512,
        device: str = "cpu",
        backend: str = "auto",
    ):
        self.model_dir = Path(model_dir) if model_dir else None
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.device = device
        self.backend = backend
        self.model: Any = None
        self.tokenizer: Any = None
        self.pipeline: Any = None

        if self.model_dir:
            self.load(str(self.model_dir))
        else:
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

    def predict(self, text: str) -> Tuple[str, float]:
        results = self.predict_batch([text])
        return results[0]

    def predict_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        if self.backend == "sklearn":
            return self._predict_batch_sklearn(texts)
        if self.backend == "paddle":
            return self._predict_batch_paddle(texts)
        raise RuntimeError(f"不支持的推理后端: {self.backend}")

    def _predict_batch_sklearn(self, texts: List[str]) -> List[Tuple[str, float]]:
        if self.pipeline is None:
            raise RuntimeError("sklearn 模型未加载")

        probabilities = self.pipeline.predict_proba(texts)
        results: List[Tuple[str, float]] = []
        for row in probabilities:
            pred_id = int(row.argmax())
            results.append((self.LABEL_MAP[pred_id], float(row[pred_id])))
        return results

    def _predict_batch_paddle(self, texts: List[str]) -> List[Tuple[str, float]]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("paddle 模型未加载")

        import paddle

        self.model.eval()
        results: List[Tuple[str, float]] = []
        with paddle.no_grad():
            for text in texts:
                encoded = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pd",
                )
                logits = self.model(**encoded)[0]
                probs = paddle.nn.functional.softmax(logits, axis=-1).numpy()[0]
                pred_id = int(probs.argmax())
                results.append((self.LABEL_MAP[pred_id], float(probs[pred_id])))
        return results

    def save(self, save_dir: str) -> None:
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)

        config = {
            "backend": self.backend,
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "max_length": self.max_length,
            "device": self.device,
        }
        with (path / "config.json").open("w", encoding="utf-8") as file:
            json.dump(config, file, ensure_ascii=False, indent=2)

        if self.backend == "sklearn":
            if self.pipeline is None:
                raise RuntimeError("没有可保存的 sklearn 模型")
            import joblib

            joblib.dump(self.pipeline, path / "model.joblib")
            return

        if self.backend == "paddle":
            if self.model is None or self.tokenizer is None:
                raise RuntimeError("没有可保存的 paddle 模型")
            self.model.save_pretrained(str(path))
            self.tokenizer.save_pretrained(str(path))
            return

        raise RuntimeError(f"未知后端: {self.backend}")

    def load(self, model_dir: str) -> None:
        path = Path(model_dir)
        config_path = path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"模型配置不存在: {config_path}")

        with config_path.open("r", encoding="utf-8") as file:
            config: Dict[str, Any] = json.load(file)

        self.backend = config.get("backend", "sklearn")
        self.model_name = config.get("model_name", self.model_name)
        self.num_labels = int(config.get("num_labels", self.num_labels))
        self.max_length = int(config.get("max_length", self.max_length))
        self.device = config.get("device", self.device)

        if self.backend == "sklearn":
            import joblib

            self.pipeline = joblib.load(path / "model.joblib")
            logger.info("已加载 sklearn 模型: {}", path)
            return

        if self.backend == "paddle":
            import paddle
            from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer

            try:
                paddle.set_device(self.device)
            except Exception:
                paddle.set_device("cpu")
                self.device = "cpu"

            self.tokenizer = AutoTokenizer.from_pretrained(str(path))
            self.model = AutoModelForSequenceClassification.from_pretrained(str(path))
            logger.info("已加载 paddle 模型: {}", path)
            return

        raise RuntimeError(f"未知后端: {self.backend}")
