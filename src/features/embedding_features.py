"""句向量特征，使用sentence-transformers并支持缓存。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin

from src.data.preprocessor import TextPreprocessor


@dataclass
class EmbeddingConfig:
    model_name: str = "all-MiniLM-L6-v2"
    cache_dir: str = "results/cache/embeddings"
    batch_size: int = 64


class EmbeddingTransformer(BaseEstimator, TransformerMixin):
    """sklearn兼容的Transformer，可用于Pipeline。"""

    def __init__(self, config: EmbeddingConfig, force_recompute: bool = False):
        self.config = config
        self.force_recompute = force_recompute
        self._model: Optional[SentenceTransformer] = None
        self._cache_file = Path(config.cache_dir) / f"{config.model_name.replace('/', '_')}.npy"
        Path(config.cache_dir).mkdir(parents=True, exist_ok=True)
        self._preprocessor = TextPreprocessor()

    def _load_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.config.model_name)
        return self._model

    def _compute_embeddings(self, texts: List[str]) -> np.ndarray:
        model = self._load_model()
        embeddings = model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        return embeddings

    def fit(self, X, y=None):
        return self

    def transform(self, df) -> np.ndarray:
        texts = self._preprocessor.transform(df)

        if self._cache_file.exists() and not self.force_recompute:
            return np.load(self._cache_file)

        embeddings = self._compute_embeddings(texts)
        np.save(self._cache_file, embeddings)
        return embeddings
