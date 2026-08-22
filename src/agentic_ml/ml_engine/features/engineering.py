"""
Feature Engineering Engine.
Generates interaction features, logarithmic/box-cox transformations for skewed variables,
and normalized ratio features.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures


class FeatureEngineer:
    """Constructs informative interaction and transformation features."""

    @staticmethod
    def add_log_transforms(df: pd.DataFrame, skew_threshold: float = 1.0) -> pd.DataFrame:
        """Applies log1p transforms to positively skewed numeric features."""
        df_out = df.copy()
        numeric_cols = df_out.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            skewness = float(df_out[col].skew())
            if skewness > skew_threshold and (df_out[col] >= 0).all():
                df_out[f"{col}_log"] = np.log1p(df_out[col])
        return df_out

    @staticmethod
    def add_polynomial_interactions(
        df: pd.DataFrame,
        degree: int = 2,
        max_features: int = 5
    ) -> pd.DataFrame:
        """Generates cross-product interactions for top numerical features."""
        df_out = df.copy()
        numeric_cols = df_out.select_dtypes(include=[np.number]).columns[:max_features]
        if len(numeric_cols) < 2:
            return df_out

        poly = PolynomialFeatures(degree=degree, interaction_only=True, include_bias=False)
        poly_arr = poly.fit_transform(df_out[numeric_cols])
        feature_names = poly.get_feature_names_out(numeric_cols)
        
        # Only add new interaction columns
        for idx, name in enumerate(feature_names):
            if name not in df_out.columns:
                df_out[name] = poly_arr[:, idx]
                
        return df_out

    @staticmethod
    def add_ratios(df: pd.DataFrame, numerator_col: str, denominator_col: str, epsilon: float = 1e-6) -> pd.DataFrame:
        """Creates normalized ratio feature."""
        df_out = df.copy()
        if numerator_col in df_out.columns and denominator_col in df_out.columns:
            df_out[f"ratio_{numerator_col}_per_{denominator_col}"] = (
                df_out[numerator_col] / (df_out[denominator_col] + epsilon)
            )
        return df_out
