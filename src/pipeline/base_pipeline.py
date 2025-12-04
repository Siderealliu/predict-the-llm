"""轻量Pipeline封装，串起特征与模型。"""

from __future__ import annotations

from typing import Any

import numpy as np


class BasePipeline:
    def __init__(self, feature_extractor: Any, model: Any):
        self.feature_extractor = feature_extractor
        self.model = model

    def fit(self, X_df, y):
        X_feat = self.feature_extractor.fit_transform(X_df) if hasattr(self.feature_extractor, "fit_transform") else self.feature_extractor.transform(X_df)
        self.model.fit(X_feat, y)
        return self

    def predict_proba(self, X_df) -> np.ndarray:
        X_feat = self.feature_extractor.transform(X_df)
        return self.model.predict_proba(X_feat)
