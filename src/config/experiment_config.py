"""按实验组集中管理配置，方便批量运行脚本调用。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass
class ExperimentSetting:
    group: str
    name: str
    feature_type: str
    model_type: str
    hpo_method: Optional[str] = None
    n_trials: Optional[int] = None
    notes: str = ""


DEFAULT_EXPERIMENTS: List[ExperimentSetting] = [
    ExperimentSetting("A", "baseline_tfidf_lr", "tfidf_word", "lr", None, None, "A1 baseline"),
    ExperimentSetting("A", "enhanced_tfidf_lr", "tfidf_word_char", "lr", None, None, "A2 word+char"),
    ExperimentSetting("B", "grid_tfidf_lr", "tfidf_word_char", "lr", "grid", 32),
    ExperimentSetting("B", "random_tfidf_lr", "tfidf_word_char", "lr", "random", 30),
    ExperimentSetting("B", "optuna_tfidf_lr", "tfidf_word_char", "lr", "optuna", 40),
    ExperimentSetting("C", "grid_embedding_lgb", "embedding", "lgb", "grid", 54),
    ExperimentSetting("C", "random_embedding_lgb", "embedding", "lgb", "random", 30),
    ExperimentSetting("C", "optuna_embedding_lgb", "embedding", "lgb", "optuna", 60),
    ExperimentSetting("D", "optuna_tfidf_word_lr", "tfidf_word", "lr", "optuna", 50),
    ExperimentSetting("D", "optuna_tfidf_word_char_lr", "tfidf_word_char", "lr", "optuna", 50),
    ExperimentSetting("D", "optuna_embedding_lr", "embedding", "lr", "optuna", 50),
    ExperimentSetting("D", "optuna_embedding_lgb", "embedding", "lgb", "optuna", 50),
    ExperimentSetting("E", "optuna_tfidf_q_only", "tfidf_word", "lr", "optuna", 30, "question only"),
    ExperimentSetting("E", "optuna_tfidf_a_only", "tfidf_word", "lr", "optuna", 30, "answer only"),
    ExperimentSetting("E", "optuna_tfidf_qa", "tfidf_word", "lr", "optuna", 30, "question+answer"),
    ExperimentSetting("E", "optuna_stat_lgb", "statistical", "lgb", "optuna", 30, "stat features"),
    ExperimentSetting("E", "optuna_tfidf_stat_lr", "tfidf_plus_stat", "lr", "optuna", 30, "tfidf+stat"),
]


def list_experiments(group: Optional[str] = None) -> Iterable[ExperimentSetting]:
    """按组返回实验配置，不传group则返回全部。"""

    if group is None:
        return DEFAULT_EXPERIMENTS
    return [exp for exp in DEFAULT_EXPERIMENTS if exp.group == group]
