"""数据划分工具，默认使用GroupKFold避免泄露。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, train_test_split


@dataclass
class GroupKFoldSplitter:
    n_splits: int = 5
    test_size: float = 0.2
    random_state: int = 42
    group_col: str = "question_hash"

    def _ensure_group_col(self, df: pd.DataFrame) -> pd.Series:
        if self.group_col in df.columns:
            return df[self.group_col]
        return df["Question"].apply(lambda x: abs(hash(x)) % 10000)

    def split(self, df: pd.DataFrame) -> Generator[Tuple[Optional[np.ndarray], np.ndarray], None, None]:
        """yield训练/验证索引，最后一个yield返回(None, holdout_idx)。"""

        df = df.reset_index(drop=True).copy()
        groups = self._ensure_group_col(df)

        train_df, holdout_df = train_test_split(
            df,
            test_size=self.test_size,
            stratify=df["target"],
            random_state=self.random_state,
        )

        gkf = GroupKFold(n_splits=self.n_splits)
        for train_idx, val_idx in gkf.split(train_df, train_df["target"], groups.loc[train_df.index]):
            yield train_idx, val_idx

        yield None, holdout_df.index.to_numpy()
