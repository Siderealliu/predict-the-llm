"""Embedding + 模型 Pipeline。"""

from __future__ import annotations

from typing import Dict

from sklearn.linear_model import LogisticRegression

from src.features.embedding_features import EmbeddingConfig, EmbeddingTransformer
from src.models.lightgbm_model import LightGBMModel
from src.pipeline.base_pipeline import BasePipeline


def build_embedding_lr_pipeline(lr_params: Dict, embed_config: EmbeddingConfig | None = None, force_recompute: bool = False) -> BasePipeline:
    embed_config = embed_config or EmbeddingConfig()
    feature = EmbeddingTransformer(embed_config, force_recompute=force_recompute)
    model = LogisticRegression(**lr_params)
    return BasePipeline(feature, model)


def build_embedding_lgb_pipeline(lgb_params: Dict, embed_config: EmbeddingConfig | None = None, force_recompute: bool = False) -> BasePipeline:
    embed_config = embed_config or EmbeddingConfig()
    feature = EmbeddingTransformer(embed_config, force_recompute=force_recompute)
    model = LightGBMModel(lgb_params)
    return BasePipeline(feature, model)
