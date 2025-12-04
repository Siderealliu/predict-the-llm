"""LightGBM多分类封装。"""

from __future__ import annotations

from typing import Any, Dict

import lightgbm as lgb

from src.models.base_model import BaseModel


class LightGBMModel(BaseModel):
    def __init__(self, params: Dict[str, Any]):
        estimator = lgb.LGBMClassifier(**params)
        super().__init__(estimator)
