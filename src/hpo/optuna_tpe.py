"""Optuna TPE 封装。"""

from __future__ import annotations

from typing import Any, Dict, Optional

import optuna
from sklearn.model_selection import cross_val_score
from sklearn.base import clone

from src.hpo.base_hpo import HPOBase


class OptunaTPEHPO(HPOBase):
    def __init__(self, estimator, cv, scoring: str = "neg_log_loss", sampler_seed: int = 42):
        super().__init__(estimator, cv, scoring)
        self.sampler_seed = sampler_seed

    def _suggest(self, trial: optuna.Trial, name: str, spec: Any):
        if isinstance(spec, list):
            return trial.suggest_categorical(name, spec)
        if len(spec) == 3 and spec[2] == "log":
            return trial.suggest_float(name, spec[0], spec[1], log=True)
        if len(spec) == 3:
            return trial.suggest_int(name, spec[0], spec[1], step=spec[2])
        if len(spec) == 2:
            if isinstance(spec[0], int) and isinstance(spec[1], int):
                return trial.suggest_int(name, spec[0], spec[1])
            return trial.suggest_float(name, spec[0], spec[1])
        raise ValueError(f"Unsupported search space entry for {name}: {spec}")

    def optimize(self, X, y, search_space: Dict[str, Any], n_trials: Optional[int] = None) -> Dict[str, Any]:
        def objective(trial: optuna.Trial):
            params = {k: self._suggest(trial, k, v) for k, v in search_space.items()}
            estimator = clone(self.estimator)
            estimator.set_params(**params)
            scores = cross_val_score(estimator, X, y, cv=self.cv, scoring=self.scoring, n_jobs=-1)
            return scores.mean()

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.sampler_seed),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        )
        study.optimize(objective, n_trials=n_trials or 20, n_jobs=1)

        return {
            "best_score": study.best_value,
            "best_params": study.best_params,
            "study": study,
        }
