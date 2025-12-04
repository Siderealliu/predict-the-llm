"""实验组C：Embedding + LightGBM 的三种HPO对比。"""

from __future__ import annotations

import time
from typing import Dict

import optuna
from sklearn.model_selection import GroupKFold, GridSearchCV, RandomizedSearchCV, cross_val_score
from scipy.stats import randint, loguniform, uniform

from src.config.data_config import get_default_data_config
from src.config.model_config import LGB_BASE_PARAMS
from src.data.data_loader import basic_preprocessing, load_data
from src.data.splitter import GroupKFoldSplitter
from src.features.embedding_features import EmbeddingConfig, EmbeddingTransformer
from src.models.lightgbm_model import LightGBMModel
from src.utils.io_utils import save_json, timestamped_filename
from src.evaluation.metrics import multiclass_logloss


def build_pipeline(lgb_params: Dict, embed_config: EmbeddingConfig | None = None, force_recompute: bool = False):
    feature = EmbeddingTransformer(embed_config or EmbeddingConfig(), force_recompute=force_recompute)
    model = LightGBMModel({**LGB_BASE_PARAMS, **lgb_params})
    return feature, model


def _get_cv(train_df):
    _, split_cfg = get_default_data_config()
    splitter = GroupKFoldSplitter(
        n_splits=split_cfg.n_splits,
        test_size=split_cfg.test_size,
        random_state=split_cfg.random_state,
        group_col=split_cfg.group_col,
    )
    folds = list(splitter.split(train_df))
    return folds[:-1], folds[-1][1]


def run_grid(train_df, cv):
    param_grid = {
        "num_leaves": [31, 63, 127],
        "max_depth": [5, 7, -1],
        "learning_rate": [0.05, 0.1, 0.2],
        "n_estimators": [200, 500],
    }
    feature, model = build_pipeline({}, force_recompute=False)
    # 预计算embedding
    X_emb = feature.transform(train_df)
    search = GridSearchCV(
        estimator=model.estimator,
        param_grid=param_grid,
        cv=cv,
        scoring="neg_log_loss",
        n_jobs=-1,
    )
    search.fit(X_emb, train_df["target"])
    return search.best_params_, search.best_score_


def run_random(train_df, cv, n_iter: int):
    param_dist = {
        "num_leaves": randint(16, 255),
        "max_depth": [-1, 4, 6, 8, 10],
        "learning_rate": loguniform(1e-3, 0.3),
        "n_estimators": randint(100, 1000),
        "min_data_in_leaf": randint(10, 200),
        "feature_fraction": uniform(0.6, 0.4),
        "lambda_l1": loguniform(1e-4, 10),
        "lambda_l2": loguniform(1e-4, 10),
    }
    feature, model = build_pipeline({}, force_recompute=False)
    X_emb = feature.transform(train_df)
    search = RandomizedSearchCV(
        estimator=model.estimator,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring="neg_log_loss",
        n_jobs=-1,
        random_state=42,
    )
    search.fit(X_emb, train_df["target"])
    return search.best_params_, search.best_score_


def run_optuna(train_df, cv, n_trials: int):
    def objective(trial: optuna.Trial):
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 16, 255),
            "max_depth": trial.suggest_categorical("max_depth", [-1, 4, 6, 8, 10]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 200),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-4, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-4, 10.0, log=True),
        }
        feature, model = build_pipeline(params, force_recompute=False)
        X_emb = feature.transform(train_df)
        scores = cross_val_score(model.estimator, X_emb, train_df["target"], cv=cv, scoring="neg_log_loss", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42), pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    return study.best_params, study.best_value, study


def run_experiment(method: str = "grid", n_trials: int | None = None, output_dir: str = "results/experiments/group_c", force_recompute: bool = False):
    paths, _ = get_default_data_config()
    train_df, _ = load_data(paths)
    train_df = basic_preprocessing(train_df)
    cv_folds, holdout_idx = _get_cv(train_df)
    cv = [(tr, va) for tr, va in cv_folds]

    start = time.time()
    if method == "grid":
        best_params, best_score = run_grid(train_df, cv)
    elif method == "random":
        best_params, best_score = run_random(train_df, cv, n_iter=n_trials or 30)
    elif method == "optuna":
        best_params, best_score, study = run_optuna(train_df, cv, n_trials=n_trials or 60)
    else:
        raise ValueError("method must be grid/random/optuna")
    elapsed = time.time() - start

    # holdout评估
    feature, model = build_pipeline(best_params, force_recompute=force_recompute)
    X_all = feature.transform(train_df)
    model.fit(X_all[~train_df.index.isin(holdout_idx)], train_df.drop(index=holdout_idx)["target"])
    holdout_pred = model.predict_proba(X_all[train_df.index.isin(holdout_idx)])
    holdout_score = multiclass_logloss(train_df.loc[holdout_idx]["target"], holdout_pred)

    results = {
        "experiment_group": "C",
        "method": method,
        "n_trials": n_trials,
        "best_score_cv": best_score,
        "holdout_logloss": holdout_score,
        "best_params": best_params,
        "elapsed_seconds": elapsed,
    }

    output_path = timestamped_filename(f"group_c_{method}", "json", output_dir)
    save_json(results, output_path)
    return results


if __name__ == "__main__":
    res = run_experiment(method="random", n_trials=5, force_recompute=True)
    print(res)
