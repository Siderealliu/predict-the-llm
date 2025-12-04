"""模型包装基类，提供统一接口。"""

from __future__ import annotations

from abc import ABC
from typing import Any


class BaseModel(ABC):
    def __init__(self, estimator: Any):
        self.estimator = estimator

    def fit(self, X, y):
        return self.estimator.fit(X, y)

    def predict(self, X):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)
