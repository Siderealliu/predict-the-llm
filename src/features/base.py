"""特征提取器基类，约定fit_transform/transform接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Protocol

import numpy as np


class FeatureExtractor(ABC):
    @abstractmethod
    def fit_transform(self, texts: Iterable[str]) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def transform(self, texts: Iterable[str]) -> np.ndarray:
        raise NotImplementedError


class TransformerProtocol(Protocol):
    """用于mypy类型提示的Protocol。"""

    def fit_transform(self, texts: Iterable[str]):
        ...

    def transform(self, texts: Iterable[str]):
        ...
