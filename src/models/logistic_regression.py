"""Logistic Regression封装。"""

from __future__ import annotations

from typing import Any, Dict

from sklearn.linear_model import LogisticRegression

from src.models.base_model import BaseModel


class LogisticRegressionModel(BaseModel):
    def __init__(self, params: Dict[str, Any]):
        estimator = LogisticRegression(**params)
        super().__init__(estimator)
