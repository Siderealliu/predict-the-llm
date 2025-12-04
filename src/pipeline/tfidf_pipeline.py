"""TF-IDF + 模型 Pipeline 构建器。"""

from __future__ import annotations

from typing import Any, Dict

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features.tfidf_features import TfidfConfig, TfidfFeatureBuilder
from src.pipeline.base_pipeline import BasePipeline


def build_tfidf_pipeline(word_params: Dict[str, Any], char_params: Dict[str, Any] | None, use_union: bool, lr_params: Dict[str, Any]) -> BasePipeline:
    tfidf_config = TfidfConfig(word_params=word_params, char_params=char_params, use_feature_union=use_union)
    feature_pipe: Pipeline = TfidfFeatureBuilder(tfidf_config).build()
    model = LogisticRegression(**lr_params)
    return BasePipeline(feature_pipe, model)
