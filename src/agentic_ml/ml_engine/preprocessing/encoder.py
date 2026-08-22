"""
Categorical Encoding Engine.
Implements K-Fold smoothed target encoding, frequency encoding, cardinality profiling,
and leakage-safe train/test alignment.
"""

import logging
from typing import Tuple, Dict, Any, List, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

logger = logging.getLogger("agentic_ml.preprocessing.encoder")


def classify_cardinality(series: pd.Series) -> str:
    """Heuristic bucket: low (<=10), medium (11-50), high (51-1000), very_high (>1000)."""
    n = series.nunique(dropna=True)
    if n <= 10:
        return "low"
    elif n <= 50:
        return "medium"
    elif n <= 1000:
        return "high"
    return "very_high"


def fit_kfold_target_encoder(
    train_df: pd.DataFrame,
    cat_col: str,
    target_col: str,
    n_splits: int = 5,
    smoothing: float = 10.0,
    seed: int = 42
) -> Tuple[pd.Series, pd.Series, float]:
    """
    Leakage-safe out-of-fold target encoding.
    Returns (encoded_train_series, final_map, global_mean).
    """
    global_mean = float(train_df[target_col].mean())
    encoded = pd.Series(index=train_df.index, dtype=float)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr_idx, val_idx in kf.split(train_df):
        fold_train = train_df.iloc[tr_idx]
        stats = fold_train.groupby(cat_col)[target_col].agg(["mean", "count"])
        smoothed = (stats["count"] * stats["mean"] + smoothing * global_mean) / (stats["count"] + smoothing)
        encoded.iloc[val_idx] = train_df.iloc[val_idx][cat_col].map(smoothed).fillna(global_mean)

    full_stats = train_df.groupby(cat_col)[target_col].agg(["mean", "count"])
    final_map = (full_stats["count"] * full_stats["mean"] + smoothing * global_mean) / (full_stats["count"] + smoothing)

    return encoded, final_map, global_mean


def apply_target_encoder(
    df: pd.DataFrame,
    cat_col: str,
    final_map: pd.Series,
    global_mean: float
) -> pd.Series:
    """Applies fitted target encoding map to unseen test/inference data."""
    return df[cat_col].map(final_map).fillna(global_mean).astype(float)


def fit_frequency_encoder(train_df: pd.DataFrame, cat_col: str) -> pd.Series:
    """Fits frequency/count distribution mapping."""
    return train_df[cat_col].value_counts(normalize=True)


def apply_frequency_encoder(df: pd.DataFrame, cat_col: str, freq_map: pd.Series) -> pd.Series:
    """Applies fitted frequency encoding map to data."""
    return df[cat_col].map(freq_map).fillna(0.0).astype(float)


def encoding_report(original_df: pd.DataFrame, encoded_df: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
    """Cardinality vs. columns produced audit."""
    rows = []
    for col in cat_cols:
        if col not in original_df.columns:
            continue
        rows.append({
            "column": col,
            "cardinality": int(original_df[col].nunique()),
            "cardinality_bucket": classify_cardinality(original_df[col]),
            "columns_after_encoding": sum(1 for c in encoded_df.columns if str(c).startswith(col)),
        })
    report = pd.DataFrame(rows)
    return report


def align_test_columns(train_encoded: pd.DataFrame, test_encoded: pd.DataFrame) -> pd.DataFrame:
    """Aligns one-hot encoded test columns to match train columns exactly, filling 0 for unseen levels."""
    return test_encoded.reindex(columns=train_encoded.columns, fill_value=0)


class FeatureEncoder:
    """Modular encoder for converting categorical columns into numerical representation."""

    def __init__(self, method: str = "onehot"):
        self.method = method
        self.cat_cols_: List[str] = []
        self.categories_: Dict[str, List[Any]] = {}
        self.freq_maps_: Dict[str, pd.Series] = {}
        self.columns_: List[str] = []

    def fit(self, X: pd.DataFrame) -> "FeatureEncoder":
        self.cat_cols_ = X.select_dtypes(include=["object", "category"]).columns.tolist()
        self.categories_ = {}
        self.freq_maps_ = {}
        
        for col in self.cat_cols_:
            if self.method == "frequency":
                self.freq_maps_[col] = fit_frequency_encoder(X, col)
            else:
                self.categories_[col] = X[col].dropna().unique().tolist()
                
        # Fit dummy transform to record output columns
        transformed = self._transform_internal(X)
        self.columns_ = transformed.columns.tolist()
        return self

    def _transform_internal(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        if not self.cat_cols_:
            return X_out
            
        if self.method == "frequency":
            for col, fmap in self.freq_maps_.items():
                if col in X_out.columns:
                    X_out[col] = apply_frequency_encoder(X_out, col, fmap)
            return X_out

        # One-hot encoding
        X_out = pd.get_dummies(X_out, columns=[c for c in self.cat_cols_ if c in X_out.columns], drop_first=False)
        return X_out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        transformed = self._transform_internal(X)
        if self.columns_:
            return transformed.reindex(columns=self.columns_, fill_value=0)
        return transformed

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)
