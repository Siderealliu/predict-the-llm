"""实验组E：消融与统计特征。"""

from __future__ import annotations

import time
from typing import Dict

import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config.data_config import get_default_data_config
from src.config.model_config import LR_BASE_PARAMS, LGB_BASE_PARAMS
from src.data.data_loader import basic_preprocessing, load_data
from src.data.splitter import GroupKFoldSplitter
from src.data.preprocessor import TextPreprocessor
from src.features.statistical_features import StatisticalFeatures
from src.models.lightgbm_model import LightGBMModel
from src.utils.io_utils import save_json, timestamped_filename
from src.evaluation.metrics import multiclass_logloss


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


def _optuna_tfidf(train_df, cv, mode: str, n_trials: int):
    def objective(trial: optuna.Trial):
        ngram = trial.suggest_categorical("ngram_range", [(1, 1), (1, 2), (1, 3)])
        max_feat = trial.suggest_int("max_features", 20000, 60000, step=20000)
        C = trial.suggest_float("C", 1e-3, 1e2, log=True)
        class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])
        pipeline = Pipeline(
            [
                ("preprocess", TextPreprocessor(mode=mode)),
                ("tfidf", TfidfVectorizer(ngram_range=ngram, max_features=max_feat, min_df=2)),
                ("classifier", LogisticRegression(**{**LR_BASE_PARAMS, "C": C, "class_weight": class_weight})),
            ]
        )
        scores = cross_val_score(pipeline, train_df, train_df["target"], cv=cv, scoring="neg_log_loss", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    return study.best_params, study.best_value, study


def _stat_only(train_df, cv):
    model = LightGBMModel(LGB_BASE_PARAMS)
    feats = StatisticalFeatures().transform(train_df)
    scores = cross_val_score(model.estimator, feats, train_df["target"], cv=cv, scoring="neg_log_loss", n_jobs=-1)
    return scores.mean()


def _tfidf_plus_stat(train_df, cv, n_trials: int):
    def objective(trial: optuna.Trial):
        ngram = trial.suggest_categorical("ngram_range", [(1, 1), (1, 2), (1, 3)])
        max_feat = trial.suggest_int("max_features", 20000, 60000, step=20000)
        C = trial.suggest_float("C", 1e-3, 1e2, log=True)
        class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])
        pipeline = Pipeline(
            [
                (
                    "features",
                    FeatureUnion(
                        [
                            ("tfidf", Pipeline([("prep", TextPreprocessor()), ("vec", TfidfVectorizer(ngram_range=ngram, max_features=max_feat, min_df=2))])),
                            ("stat", StatisticalFeatures()),
                        ]
                    ),
                ),
                ("classifier", LogisticRegression(**{**LR_BASE_PARAMS, "C": C, "class_weight": class_weight})),
            ]
        )
        scores = cross_val_score(pipeline, train_df, train_df["target"], cv=cv, scoring="neg_log_loss", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    return study.best_params, study.best_value, study


def run_experiment(output_dir: str = "results/experiments/group_e", n_trials: int = 30):
    paths, _ = get_default_data_config()
    train_df, _ = load_data(paths)
    train_df = basic_preprocessing(train_df)
    cv_folds, holdout_idx = _get_cv(train_df)
    cv = [(tr, va) for tr, va in cv_folds]

    start = time.time()
    results = {}
    # E1-E3
    for mode, tag in [("q", "q_only"), ("a", "a_only"), ("qa", "qa_concat")]:
        best_params, best_score, _ = _optuna_tfidf(train_df, cv, mode=mode, n_trials=n_trials)
        pipeline = Pipeline(
            [
                ("preprocess", TextPreprocessor(mode=mode)),
                ("tfidf", TfidfVectorizer(ngram_range=best_params["ngram_range"], max_features=best_params["max_features"], min_df=2)),
                ("classifier", LogisticRegression(**{**LR_BASE_PARAMS, "C": best_params["C"], "class_weight": best_params["class_weight"]})),
            ]
        )
        pipeline.fit(train_df.drop(index=holdout_idx), train_df.drop(index=holdout_idx)["target"])
        holdout_pred = pipeline.predict_proba(train_df.loc[holdout_idx])
        holdout_score = multiclass_logloss(train_df.loc[holdout_idx]["target"], holdout_pred)
        results[f"tfidf_{tag}"] = {"best_params": best_params, "cv_score": best_score, "holdout_logloss": holdout_score}

    # E4 stat only
    stat_cv = _stat_only(train_df, cv)
    stat_feats = StatisticalFeatures().transform(train_df)
    stat_model = LightGBMModel(LGB_BASE_PARAMS)
    stat_model.fit(stat_feats[~train_df.index.isin(holdout_idx)], train_df.drop(index=holdout_idx)["target"])
    stat_holdout_pred = stat_model.predict_proba(stat_feats[train_df.index.isin(holdout_idx)])
    stat_holdout_score = multiclass_logloss(train_df.loc[holdout_idx]["target"], stat_holdout_pred)
    results["stat_only"] = {"cv_score": stat_cv, "holdout_logloss": stat_holdout_score}

    # E5 tfidf + stat
    best_params, best_score, _ = _tfidf_plus_stat(train_df, cv, n_trials=n_trials)
    pipeline = Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        ("tfidf", Pipeline([("prep", TextPreprocessor()), ("vec", TfidfVectorizer(ngram_range=best_params["ngram_range"], max_features=best_params["max_features"], min_df=2))])),
                        ("stat", StatisticalFeatures()),
                    ]
                ),
            ),
            ("classifier", LogisticRegression(**{**LR_BASE_PARAMS, "C": best_params["C"], "class_weight": best_params["class_weight"]})),
        ]
    )
    pipeline.fit(train_df.drop(index=holdout_idx), train_df.drop(index=holdout_idx)["target"])
    holdout_pred = pipeline.predict_proba(train_df.loc[holdout_idx])
    holdout_score = multiclass_logloss(train_df.loc[holdout_idx]["target"], holdout_pred)
    results["tfidf_stat"] = {"best_params": best_params, "cv_score": best_score, "holdout_logloss": holdout_score}

    elapsed = time.time() - start
    output_path = timestamped_filename("group_e_ablation", "json", output_dir)
    save_json(
        {
            "experiment_group": "E",
            "results": results,
            "elapsed_seconds": elapsed,
        },
        output_path,
    )
    return results


if __name__ == "__main__":
    res = run_experiment(n_trials=5)
    print(res)
