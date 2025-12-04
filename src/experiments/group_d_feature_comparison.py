"""实验组D：四种特征+模型组合，统一Optuna对比。"""

from __future__ import annotations

import time
from typing import Dict, Tuple

import numpy as np
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from src.config.data_config import get_default_data_config
from src.config.model_config import LR_BASE_PARAMS, LGB_BASE_PARAMS, get_lgb_params
from src.data.data_loader import basic_preprocessing, load_data
from src.data.splitter import GroupKFoldSplitter
from src.features.embedding_features import EmbeddingConfig, EmbeddingTransformer
from src.features.tfidf_features import TfidfConfig, TfidfFeatureBuilder
from src.features.statistical_features import StatisticalFeatures
from src.models.lightgbm_model import LightGBMModel
from src.utils.io_utils import save_json, timestamped_filename
from src.evaluation.metrics import multiclass_logloss


def _use_gpu_flag() -> bool:
    import os

    return os.environ.get("USE_GPU", "").lower() in ("1", "true", "yes")


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


def _pipeline_tfidf(word_ngram, max_features, use_char=False, char_ngram=(3, 5)):
    word_params = {"ngram_range": word_ngram, "max_features": max_features, "min_df": 2}
    char_params = {"analyzer": "char", "ngram_range": char_ngram, "max_features": max_features, "min_df": 2} if use_char else None
    tfidf = TfidfFeatureBuilder(TfidfConfig(word_params=word_params, char_params=char_params, use_feature_union=use_char)).build()
    return tfidf


def _objective_tfidf_lr(train_df, cv, use_char: bool):
    def objective(trial: optuna.Trial):
        ngram = trial.suggest_categorical("word_ngram", [(1, 1), (1, 2), (1, 3)])
        max_feat = trial.suggest_int("max_features", 20000, 60000, step=20000)
        if use_char:
            char_ngram = trial.suggest_categorical("char_ngram", [(3, 5), (3, 6)])
        tfidf_pipe = _pipeline_tfidf(ngram, max_feat, use_char=use_char, char_ngram=char_ngram if use_char else (3, 5))
        C = trial.suggest_float("C", 1e-3, 1e2, log=True)
        class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])
        estimator = tfidf_pipe
        estimator.steps.append(("classifier", LogisticRegression(**{**LR_BASE_PARAMS, "C": C, "class_weight": class_weight})))
        scores = cross_val_score(estimator, train_df, train_df["target"], cv=cv, scoring="neg_log_loss", n_jobs=-1)
        return scores.mean()

    return objective


def _objective_embedding_lr(train_df, cv, embeddings: np.ndarray):
    def objective(trial: optuna.Trial):
        C = trial.suggest_float("C", 1e-3, 1e2, log=True)
        class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])
        model = LogisticRegression(**{**LR_BASE_PARAMS, "C": C, "class_weight": class_weight})
        scores = cross_val_score(model, embeddings, train_df["target"], cv=cv, scoring="neg_log_loss", n_jobs=-1)
        return scores.mean()

    return objective


def _objective_embedding_lgb(train_df, cv, embeddings: np.ndarray):
    def objective(trial: optuna.Trial):
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 16, 255),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        }
        model = LightGBMModel({**get_lgb_params(use_gpu=_use_gpu_flag()), **params})
        scores = cross_val_score(model.estimator, embeddings, train_df["target"], cv=cv, scoring="neg_log_loss", n_jobs=-1)
        return scores.mean()

    return objective


def run_experiment(feature_model: str, n_trials: int = 50, output_dir: str = "results/experiments/group_d"):
    paths, _ = get_default_data_config()
    train_df, _ = load_data(paths)
    train_df = basic_preprocessing(train_df)
    cv_folds, holdout_idx = _get_cv(train_df)
    cv = [(tr, va) for tr, va in cv_folds]

    # 如果需要embedding，先预计算一次
    embeddings = None
    if "embedding" in feature_model:
        embed = EmbeddingTransformer(EmbeddingConfig(), force_recompute=False)
        embeddings = embed.transform(train_df)

    if feature_model == "tfidf_word_lr":
        objective = _objective_tfidf_lr(train_df, cv, use_char=False)
    elif feature_model == "tfidf_word_char_lr":
        objective = _objective_tfidf_lr(train_df, cv, use_char=True)
    elif feature_model == "embedding_lr":
        objective = _objective_embedding_lr(train_df, cv, embeddings)
    elif feature_model == "embedding_lgb":
        objective = _objective_embedding_lgb(train_df, cv, embeddings)
    else:
        raise ValueError("feature_model not supported")

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42), pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
    start = time.time()
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    elapsed = time.time() - start

    best_params = study.best_params
    best_score = study.best_value

    # holdout评估
    if feature_model.startswith("tfidf"):
        ngram = best_params.get("word_ngram", (1, 2))
        max_feat = best_params.get("max_features", 40000)
        use_char = feature_model == "tfidf_word_char_lr"
        char_ngram = best_params.get("char_ngram", (3, 5))
        tfidf_pipe = _pipeline_tfidf(ngram, max_feat, use_char=use_char, char_ngram=char_ngram)
        model = LogisticRegression(**{**LR_BASE_PARAMS, "C": best_params.get("C", 1.0), "class_weight": best_params.get("class_weight")})
        estimator = tfidf_pipe
        estimator.steps.append(("classifier", model))
        estimator.fit(train_df.drop(index=holdout_idx), train_df.drop(index=holdout_idx)["target"])
        holdout_pred = estimator.predict_proba(train_df.loc[holdout_idx])
    elif feature_model == "embedding_lr":
        model = LogisticRegression(**{**LR_BASE_PARAMS, "C": best_params.get("C", 1.0), "class_weight": best_params.get("class_weight")})
        model.fit(embeddings[~train_df.index.isin(holdout_idx)], train_df.drop(index=holdout_idx)["target"])
        holdout_pred = model.predict_proba(embeddings[train_df.index.isin(holdout_idx)])
    else:  # embedding_lgb
        params = {
            "num_leaves": best_params.get("num_leaves", 63),
            "learning_rate": best_params.get("learning_rate", 0.1),
            "n_estimators": best_params.get("n_estimators", 500),
        }
        model = LightGBMModel({**get_lgb_params(use_gpu=_use_gpu_flag()), **params})
        model.fit(embeddings[~train_df.index.isin(holdout_idx)], train_df.drop(index=holdout_idx)["target"])
        holdout_pred = model.predict_proba(embeddings[train_df.index.isin(holdout_idx)])

    holdout_score = multiclass_logloss(train_df.loc[holdout_idx]["target"], holdout_pred)

    results = {
        "experiment_group": "D",
        "feature_model": feature_model,
        "n_trials": n_trials,
        "best_score_cv": best_score,
        "holdout_logloss": holdout_score,
        "best_params": best_params,
        "elapsed_seconds": elapsed,
    }

    output_path = timestamped_filename(f"group_d_{feature_model}", "json", output_dir)
    save_json(results, output_path)
    return results


if __name__ == "__main__":
    res = run_experiment("tfidf_word_lr", n_trials=5)
    print(res)
