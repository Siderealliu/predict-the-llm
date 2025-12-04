"""实验组A：TF-IDF + LR 基线。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from src.config.data_config import get_default_data_config
from src.config.model_config import LR_BASE_PARAMS, TFIDF_WORD_PARAMS_DEFAULT, TFIDF_CHAR_PARAMS_DEFAULT
from src.data.data_loader import basic_preprocessing, load_data
from src.data.splitter import GroupKFoldSplitter
from src.evaluation.metrics import multiclass_logloss
from src.features.tfidf_features import TfidfFeatureBuilder, TfidfConfig
from src.utils.io_utils import save_json, timestamped_filename


def run_baseline(output_dir: str = "results/experiments/group_a") -> Dict:
    paths, split_cfg = get_default_data_config()
    train_df, _ = load_data(paths)
    train_df = basic_preprocessing(train_df)

    splitter = GroupKFoldSplitter(
        n_splits=split_cfg.n_splits,
        test_size=split_cfg.test_size,
        random_state=split_cfg.random_state,
        group_col=split_cfg.group_col,
    )

    fold_indices = list(splitter.split(train_df))
    holdout_idx = fold_indices[-1][1]
    cv_folds = fold_indices[:-1]

    tfidf_pipeline = TfidfFeatureBuilder(
        TfidfConfig(word_params=TFIDF_WORD_PARAMS_DEFAULT, char_params=None, use_feature_union=False)
    ).build()
    estimator = Pipeline(tfidf_pipeline.steps + [("classifier", LogisticRegression(**LR_BASE_PARAMS))])

    gkf = GroupKFold(n_splits=split_cfg.n_splits)
    start = time.time()
    oof_proba = cross_val_predict(
        estimator,
        train_df,
        train_df["target"],
        cv=gkf.split(train_df, train_df["target"], train_df["Question"]),
        method="predict_proba",
        n_jobs=-1,
    )
    cv_score = multiclass_logloss(train_df["target"], oof_proba)

    holdout_df = train_df.loc[holdout_idx]
    estimator.fit(train_df.drop(index=holdout_idx), train_df.drop(index=holdout_idx)["target"])
    holdout_pred = estimator.predict_proba(holdout_df)
    holdout_score = multiclass_logloss(holdout_df["target"], holdout_pred)
    elapsed = time.time() - start

    results = {
        "experiment_group": "A",
        "name": "baseline_tfidf_lr",
        "cv_logloss": cv_score,
        "holdout_logloss": holdout_score,
        "elapsed_seconds": elapsed,
    }

    output_path = timestamped_filename("group_a_baseline", "json", output_dir)
    save_json(results, output_path)
    return results


if __name__ == "__main__":
    res = run_baseline()
    print(res)
