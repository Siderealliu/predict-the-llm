"""HPO 基类与统一接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class HPOBase(ABC):
    def __init__(self, estimator, cv, scoring: str = "neg_log_loss"):
        self.estimator = estimator
        self.cv = cv
        self.scoring = scoring

    @abstractmethod
    def optimize(self, X, y, search_space: Dict[str, Any], n_trials: Optional[int] = None) -> Dict[str, Any]:
        raise NotImplementedError
