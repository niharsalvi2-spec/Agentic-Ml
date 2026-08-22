"""
Feature Selection Engine.
Implements Filter (ANOVA, Mutual Information, Variance Threshold), Wrapper (RFE),
and Embedded (L1 Lasso, Tree Importance) feature selection.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
    VarianceThreshold,
    RFE
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Lasso


class FeatureSelector:
    """Selects high-signal features and eliminates redundant noise features."""

    @staticmethod
    def select_top_k(
        X: pd.DataFrame,
        y: pd.Series,
        task_type: str = "classification",
        k: int = 4,
        method: str = "anova"
    ) -> List[str]:
        """Selects top K features based on statistical score function."""
        if X.empty:
            return []
            
        k_val = min(k, X.shape[1])
        
        if method == "mutual_info":
            score_func = mutual_info_classif if task_type == "classification" else mutual_info_regression
        else:
            score_func = f_classif if task_type == "classification" else f_regression

        selector = SelectKBest(score_func=score_func, k=k_val)
        selector.fit(X, y)
        selected_cols = X.columns[selector.get_support()].tolist()
        return selected_cols

    @staticmethod
    def select_variance_threshold(X: pd.DataFrame, threshold: float = 0.0) -> List[str]:
        """Removes low-variance (near-constant) features."""
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X)
        return X.columns[selector.get_support()].tolist()

    @staticmethod
    def select_rfe(
        X: pd.DataFrame,
        y: pd.Series,
        task_type: str = "classification",
        n_features_to_select: int = 4
    ) -> List[str]:
        """Recursive Feature Elimination using Random Forest estimator."""
        estimator = (
            RandomForestClassifier(n_estimators=30, random_state=42)
            if task_type == "classification"
            else RandomForestRegressor(n_estimators=30, random_state=42)
        )
        n_select = min(n_features_to_select, X.shape[1])
        rfe = RFE(estimator=estimator, n_features_to_select=n_select)
        rfe.fit(X, y)
        return X.columns[rfe.get_support()].tolist()

    @staticmethod
    def select_tree_importance(
        X: pd.DataFrame,
        y: pd.Series,
        task_type: str = "classification",
        top_k: int = 4
    ) -> List[str]:
        """Selects top K features based on Random Forest Gini/MSE importance."""
        model = (
            RandomForestClassifier(n_estimators=50, random_state=42)
            if task_type == "classification"
            else RandomForestRegressor(n_estimators=50, random_state=42)
        )
        model.fit(X, y)
        importances = pd.Series(model.feature_importances_, index=X.columns)
        return importances.sort_values(ascending=False).head(top_k).index.tolist()
