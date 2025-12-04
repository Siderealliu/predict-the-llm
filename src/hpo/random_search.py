"""Randomized Search 封装。"""

from __future__ import annotations

from typing import Any, Dict, Optional

from sklearn.model_selection import RandomizedSearchCV

from src.hpo.base_hpo import HPOBase


class RandomSearchHPO(HPOBase):
    def optimize(self, X, y, search_space: Dict[str, Any], n_trials: Optional[int] = None) -> Dict[str, Any]:
        search = RandomizedSearchCV(
            estimator=self.estimator,
            param_distributions=search_space,
            n_iter=n_trials or 10,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=-1,
            random_state=42,
        )
        search.fit(X, y)
        return {
            "best_score": search.best_score_,
            "best_params": search.best_params_,
            "cv_results": search.cv_results_,
        }
