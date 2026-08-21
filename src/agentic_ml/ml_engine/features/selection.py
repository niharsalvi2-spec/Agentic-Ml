import pandas as pd
from typing import List
from sklearn.feature_selection import SelectKBest, f_classif, f_regression

class FeatureSelector:
    """Selects top K features based on statistical significance."""
    
    @staticmethod
    def select_top_k(X: pd.DataFrame, y: pd.Series, task_type: str = "classification", k: int = 4) -> List[str]:
        k = min(k, X.shape[1])
        score_func = f_classif if task_type == "classification" else f_regression
        selector = SelectKBest(score_func=score_func, k=k)
        selector.fit(X, y)
        selected_cols = X.columns[selector.get_support()].tolist()
        return selected_cols
