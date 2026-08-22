"""
Feature Scaling module.
Supports standard, min-max, and robust IQR-based scaling.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


class FeatureScaler:
    """Scales numerical features using fitted statistical parameters."""

    def __init__(self, method: str = "standard"):
        self.method = method
        if method == "minmax":
            self.scaler = MinMaxScaler()
        elif method == "robust":
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
            
        self.numeric_cols_: List[str] = []

    def fit(self, X: pd.DataFrame) -> "FeatureScaler":
        self.numeric_cols_ = X.select_dtypes(include=["number"]).columns.tolist()
        if self.numeric_cols_:
            self.scaler.fit(X[self.numeric_cols_])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        valid_cols = [c for c in self.numeric_cols_ if c in X_out.columns]
        if valid_cols:
            X_out[valid_cols] = self.scaler.transform(X_out[valid_cols])
        return X_out

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)
