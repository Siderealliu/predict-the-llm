"""集中保存默认模型参数和搜索空间，便于实验复用。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable


@dataclass
class SearchSpace:
    """描述一个HPO搜索空间，值通常是列表或分布描述。"""

    name: str
    space: Dict[str, Iterable[Any]]
    notes: str | None = None


# ---- 基础参数 ----
TFIDF_WORD_PARAMS_DEFAULT: Dict[str, Any] = {
    "ngram_range": (1, 2),
    "max_features": 40000,
    "min_df": 2,
}

TFIDF_CHAR_PARAMS_DEFAULT: Dict[str, Any] = {
    "analyzer": "char",
    "ngram_range": (3, 5),
    "max_features": 60000,
    "min_df": 2,
}

LR_BASE_PARAMS: Dict[str, Any] = {
    "C": 1.0,
    "class_weight": None,
    "multi_class": "auto",
    "max_iter": 200,
    "n_jobs": -1,
}

LGB_BASE_PARAMS: Dict[str, Any] = {
    "objective": "multiclass",
    "num_class": 7,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "verbose": -1,
}

# GPU 加速可选参数（启用时在构建模型时合并）
LGB_GPU_PARAMS: Dict[str, Any] = {
    "device": "gpu",
}


def get_lgb_params(use_gpu: bool = False) -> Dict[str, Any]:
    params = {**LGB_BASE_PARAMS}
    if use_gpu:
        params.update(LGB_GPU_PARAMS)
    return params


# ---- HPO 搜索空间：组B TF-IDF + LR ----
GRID_SEARCH_SPACE_B = SearchSpace(
    name="grid_tfidf_lr",
    space={
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__max_features": [20000, 40000],
        "classifier__C": [0.01, 0.1, 1.0, 10.0],
        "classifier__class_weight": [None, "balanced"],
    },
    notes="与Outline B1一致的网格搜索空间",
)

RANDOM_SEARCH_SPACE_B = SearchSpace(
    name="random_tfidf_lr",
    space={
        "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
        "tfidf__max_features": [20000, 40000, 60000],
        "classifier__C": ("loguniform", 1e-3, 1e2),
        "classifier__class_weight": [None, "balanced"],
    },
    notes="随机搜索使用loguniform标记表示连续空间",
)

OPTUNA_SEARCH_SPACE_B = SearchSpace(
    name="optuna_tfidf_lr",
    space={
        "ngram_range": [(1, 1), (1, 2), (1, 3)],
        "max_features": (20000, 60000, 20000),  # start, end, step
        "C": (1e-3, 1e2, "log"),
        "class_weight": [None, "balanced"],
    },
    notes="Optuna TPE搜索空间，逻辑由hpo模块具体解释",
)


# ---- HPO 搜索空间：组C Embedding + LightGBM ----
GRID_SEARCH_SPACE_C = SearchSpace(
    name="grid_embedding_lgb",
    space={
        "num_leaves": [31, 63, 127],
        "max_depth": [5, 7, -1],
        "learning_rate": [0.05, 0.1, 0.2],
        "n_estimators": [200, 500],
    },
)

RANDOM_SEARCH_SPACE_C = SearchSpace(
    name="random_embedding_lgb",
    space={
        "num_leaves": ("randint", 16, 255),
        "max_depth": [-1, 4, 6, 8, 10],
        "learning_rate": ("loguniform", 1e-3, 0.3),
        "n_estimators": ("randint", 100, 1000),
        "min_data_in_leaf": ("randint", 10, 200),
        "feature_fraction": ("uniform", 0.6, 0.4),
        "lambda_l1": ("loguniform", 1e-4, 10),
        "lambda_l2": ("loguniform", 1e-4, 10),
    },
)

OPTUNA_SEARCH_SPACE_C = SearchSpace(
    name="optuna_embedding_lgb",
    space={
        "num_leaves": (16, 255),
        "max_depth": [-1, 4, 6, 8, 10],
        "learning_rate": (1e-3, 0.3, "log"),
        "n_estimators": (100, 1000),
        "min_data_in_leaf": (10, 200),
        "feature_fraction": (0.6, 1.0),
        "lambda_l1": (1e-4, 10.0, "log"),
        "lambda_l2": (1e-4, 10.0, "log"),
    },
)
