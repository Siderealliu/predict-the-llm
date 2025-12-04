"""简单的统计特征提取。"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from src.data.preprocessor import TextPreprocessor


class StatisticalFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._preprocessor = TextPreprocessor()

    def fit(self, X, y=None):
        return self

    def transform(self, df):
        texts = self._preprocessor.transform(df)
        features = []
        for text in texts:
            words = text.split()
            word_lengths = [len(w) for w in words] or [0]
            char_count = len(text)
            feature_row = [
                char_count,
                len(words),
                float(np.mean(word_lengths)),
                sum(c in ".,!?;:" for c in text) / max(char_count, 1),
                sum(c.isupper() for c in text) / max(char_count, 1),
                sum(c.isdigit() for c in text) / max(char_count, 1),
                text.count(".") + text.count("!") + text.count("?"),
            ]
            features.append(feature_row)
        return np.array(features)
