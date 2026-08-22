"""
Hyperparameter Tuning Engine.
Provides GridSearchCV and RandomizedSearchCV optimization harnesses.
"""

from typing import Dict, Any, Optional
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


class HyperparameterTuner:
    """Tunes candidate models using cross-validated search."""

    @staticmethod
    def tune(
        model: Any,
        param_grid: Dict[str, list],
        X: pd.DataFrame,
        y: pd.Series,
        scoring: Optional[str] = None,
        cv: int = 3,
        n_iter: Optional[int] = None,
        random_state: int = 42
    ) -> Dict[str, Any]:
        if n_iter:
            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grid,
                n_iter=n_iter,
                scoring=scoring,
                cv=cv,
                random_state=random_state
            )
        else:
            search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                scoring=scoring,
                cv=cv
            )

        search.fit(X, y)
        return {
            "best_params": search.best_params_,
            "best_score": round(float(search.best_score_), 4),
            "best_estimator": search.best_estimator_
        }
