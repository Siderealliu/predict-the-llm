"""实验组B：TF-IDF + LR 三种HPO对比。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import loguniform

from src.config.data_config import get_default_data_config
from src.config.model_config import LR_BASE_PARAMS
from src.data.data_loader import basic_preprocessing, load_data
from src.data.splitter import GroupKFoldSplitter
from src.data.preprocessor import TextPreprocessor
from src.evaluation.metrics import multiclass_logloss
from src.utils.io_utils import save_json, timestamped_filename


def build_pipeline(word_params: Dict, char_params: Dict) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", TextPreprocessor()),
            (
                "features",
                FeatureUnion(
                    [
                        ("word_tfidf", TfidfVectorizer(**word_params)),
                        ("char_tfidf", TfidfVectorizer(**char_params)),
                    ]
                ),
            ),
            ("classifier", LogisticRegression(**LR_BASE_PARAMS)),
        ]
    )


def _split_train_holdout(train_df):
    paths, split_cfg = get_default_data_config()
    splitter = GroupKFoldSplitter(
        n_splits=split_cfg.n_splits,
        test_size=split_cfg.test_size,
        random_state=split_cfg.random_state,
        group_col=split_cfg.group_col,
    )
    folds = list(splitter.split(train_df))
    cv_folds = folds[:-1]
    holdout_idx = folds[-1][1]
    return cv_folds, holdout_idx


def run_grid(train_df, cv):
    word_params = {"ngram_range": (1, 2), "max_features": 40000, "min_df": 2}
    char_params = {"analyzer": "char", "ngram_range": (3, 5), "max_features": 60000, "min_df": 2}
    pipeline = build_pipeline(word_params, char_params)

    param_grid = {
        "features__word_tfidf__ngram_range": [(1, 1), (1, 2)],
        "features__word_tfidf__max_features": [20000, 40000],
        "classifier__C": [0.01, 0.1, 1.0, 10.0],
        "classifier__class_weight": [None, "balanced"],
    }

    search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="neg_log_loss",
        n_jobs=-1,
    )
    search.fit(train_df, train_df["target"])
    return search.best_params_, search.best_score_


def run_random(train_df, cv, n_iter: int):
    word_params = {"ngram_range": (1, 2), "max_features": 40000, "min_df": 2}
    char_params = {"analyzer": "char", "ngram_range": (3, 5), "max_features": 60000, "min_df": 2}
    pipeline = build_pipeline(word_params, char_params)

    param_dist = {
        "features__word_tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
        "features__word_tfidf__max_features": [20000, 40000, 60000],
        "classifier__C": loguniform(1e-3, 1e2),
        "classifier__class_weight": [None, "balanced"],
    }

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring="neg_log_loss",
        n_jobs=-1,
        random_state=42,
    )
    search.fit(train_df, train_df["target"])
    return search.best_params_, search.best_score_


def run_optuna(train_df, cv, n_trials: int):
    word_params = {"ngram_range": (1, 2), "max_features": 40000, "min_df": 2}
    char_params = {"analyzer": "char", "ngram_range": (3, 5), "max_features": 60000, "min_df": 2}

    def objective(trial: optuna.Trial):
        wp = word_params.copy()
        wp["ngram_range"] = trial.suggest_categorical("word_ngram_range", [(1, 1), (1, 2), (1, 3)])
        wp["max_features"] = trial.suggest_int("word_max_features", 20000, 60000, step=20000)
        cp = char_params.copy()
        C = trial.suggest_float("C", 1e-3, 1e2, log=True)
        class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])

        pipeline = build_pipeline(wp, cp)
        pipeline.set_params(classifier__C=C, classifier__class_weight=class_weight)
        scores = cross_val_score(pipeline, train_df, train_df["target"], cv=cv, scoring="neg_log_loss", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    return study.best_params, study.best_value, study


def run_experiment(method: str = "grid", n_trials: int | None = None, output_dir: str = "results/experiments/group_b"):
    paths, _ = get_default_data_config()
    train_df, _ = load_data(paths)
    train_df = basic_preprocessing(train_df)
    cv_folds, holdout_idx = _split_train_holdout(train_df)
    cv = [(tr, va) for tr, va in cv_folds]

    start = time.time()
    if method == "grid":
        best_params, best_score = run_grid(train_df, cv)
    elif method == "random":
        best_params, best_score = run_random(train_df, cv, n_iter=n_trials or 30)
    elif method == "optuna":
        best_params, best_score, study = run_optuna(train_df, cv, n_trials=n_trials or 40)
    else:
        raise ValueError("method must be one of grid/random/optuna")
    elapsed = time.time() - start

    # holdout评估
    # 兼容不同搜索方式返回的参数名
    word_ngram = best_params.get("features__word_tfidf__ngram_range") or best_params.get("word_ngram_range") or (1, 2)
    word_max_feat = best_params.get("features__word_tfidf__max_features") or best_params.get("word_max_features") or 40000
    word_params = {"ngram_range": word_ngram, "max_features": word_max_feat, "min_df": 2}
    char_params = {"analyzer": "char", "ngram_range": (3, 5), "max_features": 60000, "min_df": 2}
    pipeline = build_pipeline(word_params, char_params)
    if "classifier__C" in best_params:
        pipeline.set_params(**{k: v for k, v in best_params.items() if k.startswith("classifier__")})
    else:
        pipeline.set_params(classifier__C=best_params.get("C", 1.0), classifier__class_weight=best_params.get("class_weight", None))

    pipeline.fit(train_df.drop(index=holdout_idx), train_df.drop(index=holdout_idx)["target"])
    holdout_pred = pipeline.predict_proba(train_df.loc[holdout_idx])
    holdout_score = multiclass_logloss(train_df.loc[holdout_idx]["target"], holdout_pred)

    results = {
        "experiment_group": "B",
        "method": method,
        "n_trials": n_trials,
        "best_score_cv": best_score,
        "holdout_logloss": holdout_score,
        "best_params": best_params,
        "elapsed_seconds": elapsed,
    }

    output_path = timestamped_filename(f"group_b_{method}", "json", output_dir)
    save_json(results, output_path)
    return results


if __name__ == "__main__":
    res = run_experiment(method="grid", n_trials=32)
    print(res)
