"""TF-IDF特征提取器，支持word和char两路合并。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline

from src.data.preprocessor import TextPreprocessor


@dataclass
class TfidfConfig:
    word_params: Dict[str, Any]
    char_params: Optional[Dict[str, Any]] = None
    use_feature_union: bool = False


class TfidfFeatureBuilder:
    """构建带预处理的TF-IDF流水线。"""

    def __init__(self, config: TfidfConfig):
        self.config = config

    def build(self) -> Pipeline:
        preproc = TextPreprocessor()

        if self.config.use_feature_union and self.config.char_params:
            features = FeatureUnion(
                [
                    ("word_tfidf", TfidfVectorizer(**self.config.word_params)),
                    ("char_tfidf", TfidfVectorizer(**self.config.char_params)),
                ]
            )
            pipeline = Pipeline(
                [
                    ("preprocess", preproc),
                    ("features", features),
                ]
            )
        else:
            pipeline = Pipeline(
                [
                    ("preprocess", preproc),
                    ("tfidf", TfidfVectorizer(**self.config.word_params)),
                ]
            )
        return pipeline


def transform_texts(pipeline: Pipeline, df) -> np.ndarray:
    texts: List[str] = pipeline.named_steps["preprocess"].transform(df)
    vec_step = pipeline.named_steps.get("features") or pipeline.named_steps.get("tfidf")
    return vec_step.transform(texts)
