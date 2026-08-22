"""
Data Imputation module.
Supports mean, median, mode, and constant imputation strategies with fit/transform separation.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


class DataImputer:
    """Imputes missing values in numerical and categorical features."""

    def __init__(self, numeric_strategy: str = "median", categorical_strategy: str = "mode"):
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.statistics_: Dict[str, Any] = {}

    def fit(self, X: pd.DataFrame) -> "DataImputer":
        self.statistics_ = {}
        for col in X.columns:
            series = X[col].dropna()
            if series.empty:
                self.statistics_[col] = 0 if pd.api.types.is_numeric_dtype(X[col]) else "missing"
                continue

            if pd.api.types.is_numeric_dtype(X[col]):
                if self.numeric_strategy == "mean":
                    self.statistics_[col] = float(series.mean())
                else:
                    self.statistics_[col] = float(series.median())
            else:
                # Mode for categorical
                mode_vals = series.mode()
                self.statistics_[col] = mode_vals.iloc[0] if not mode_vals.empty else "missing"
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        for col, val in self.statistics_.items():
            if col in X_out.columns:
                X_out[col] = X_out[col].fillna(val)
        return X_out

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)
