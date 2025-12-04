"""数据加载与基础清洗。"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from src.config.data_config import DataPaths


def load_data(paths: DataPaths | None = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """读取train/test数据集。"""

    paths = paths or DataPaths()
    train_df = pd.read_csv(Path(paths.train_path))
    test_df = pd.read_csv(Path(paths.test_path))
    return train_df, test_df


def basic_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """基础清洗：去除完全缺失行并重置索引。"""

    cleaned = df.dropna(subset=["Question", "Response"])
    cleaned = cleaned.reset_index(drop=True)
    return cleaned
