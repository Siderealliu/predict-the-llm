"""Grid Search 封装。"""

from __future__ import annotations

from typing import Any, Dict, Optional

from sklearn.model_selection import GridSearchCV

from src.hpo.base_hpo import HPOBase


class GridSearchHPO(HPOBase):
    def optimize(self, X, y, search_space: Dict[str, Any], n_trials: Optional[int] = None) -> Dict[str, Any]:
        grid = GridSearchCV(
            estimator=self.estimator,
            param_grid=search_space,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=-1,
        )
        grid.fit(X, y)
        grid_results = grid
        return {
            "best_score": grid_results.best_score_,
            "best_params": grid_results.best_params_,
            "cv_results": grid_results.cv_results_,
        }
