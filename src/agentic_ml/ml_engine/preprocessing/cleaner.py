"""
Data Cleaning & Preprocessing Engine.
Implements leakage-safe cleaning, Tukey IQR outlier fences, MAD-based modified Z-scores,
completeness-based deduplication, and MCAR/MAR/MNAR statistical diagnostics.
"""

import logging
from typing import Tuple, Dict, Any, Optional, List
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("agentic_ml.preprocessing.cleaner")


def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column missing count and percentage."""
    report = pd.DataFrame({
        "missing_count": df.isnull().sum(),
        "missing_pct": (df.isnull().sum() / max(1, len(df)) * 100).round(2),
        "dtype": df.dtypes.astype(str),
    }).sort_values("missing_pct", ascending=False)
    return report


def compute_iqr_fences(series: pd.Series, k: float = 1.5) -> Tuple[float, float]:
    """Compute Tukey IQR fences on a training series. Returns (lower, upper)."""
    clean_series = series.dropna()
    if len(clean_series) < 2:
        return float("-inf"), float("inf")
    q1 = float(clean_series.quantile(0.25))
    q3 = float(clean_series.quantile(0.75))
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def apply_fences(df: pd.DataFrame, fences: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    """Winsorize (clip) columns to pre-computed fences."""
    df_out = df.copy()
    for col, (lower, upper) in fences.items():
        if col in df_out.columns:
            df_out[col] = df_out[col].clip(lower, upper)
    return df_out


def modified_z_outliers(series: pd.Series, threshold: float = 3.5) -> pd.Series:
    """Boolean mask of outliers via MAD-based modified Z-score (robust to skew)."""
    clean_series = series.dropna()
    if len(clean_series) < 2:
        return pd.Series(False, index=series.index)
    median = clean_series.median()
    diff = (clean_series - median).abs()
    mad = diff.median()
    if mad == 0:
        # Fallback to mean absolute deviation for tied data
        mad = diff.mean()
    if mad == 0:
        return pd.Series(False, index=series.index)
    modified_z = 0.6745 * (series - median) / mad
    return modified_z.abs() > threshold


def dedupe_with_log(df: pd.DataFrame, subset: Optional[List[str]] = None, keep: str = "first") -> pd.DataFrame:
    """Drop duplicates, preferring the most complete row per key, with logged count."""
    before_n = len(df)
    df_temp = df.copy()
    df_temp["_completeness"] = df_temp.notnull().sum(axis=1)
    if subset:
        df_temp = (df_temp.sort_values("_completeness", ascending=False)
                   .drop_duplicates(subset=subset, keep="first"))
    else:
        df_temp = df_temp.drop_duplicates(keep=keep)
    df_cleaned = df_temp.drop(columns="_completeness", errors="ignore")
    logger.info(f"Dedup: {before_n} -> {len(df_cleaned)} rows ({before_n - len(df_cleaned)} removed)")
    return df_cleaned


def cleaning_report(before: pd.DataFrame, after: pd.DataFrame, label: str = "") -> pd.DataFrame:
    """Before/after row count + missingness diff for a cleaning step."""
    rows = []
    for col in before.columns:
        if col not in after.columns:
            continue
        rows.append({
            "column": col,
            "missing_before": int(before[col].isnull().sum()),
            "missing_after": int(after[col].isnull().sum()),
        })
    report = pd.DataFrame(rows)
    if label:
        logger.info(f"Cleaning report: {label} ({len(before)} -> {len(after)} rows)")
    return report


def classify_missingness_hint(
    df: pd.DataFrame,
    target_col: str,
    other_cols: Optional[List[str]] = None,
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    MCAR-vs-not signal: t-test comparing other_cols between rows where target_col
    is missing vs not. Significant differences suggest MAR/MNAR rather than MCAR.
    """
    if other_cols is None:
        other_cols = df.select_dtypes(include="number").columns.drop(
            target_col, errors="ignore"
        ).tolist()
        
    is_missing = df[target_col].isnull()
    results = {}
    for col in other_cols:
        if col == target_col or col not in df.columns:
            continue
        a = df.loc[is_missing, col].dropna()
        b = df.loc[~is_missing, col].dropna()
        if len(a) < 2 or len(b) < 2:
            continue
        _, p = stats.ttest_ind(a, b, equal_var=False)
        results[col] = {"p_value": round(float(p), 4), "likely_not_MCAR": bool(p < alpha)}
    return pd.DataFrame(results).T


class DeterministicPreprocessor:
    """Handles missing value imputation, winsorization, and scaling with train/test isolation."""

    def __init__(self, clip_outliers: bool = False):
        self.scaler = StandardScaler()
        self.clip_outliers = clip_outliers
        self.fences: Dict[str, Tuple[float, float]] = {}
        self.medians: Dict[str, float] = {}
        self.numeric_cols: List[str] = []

    def fit(self, X: pd.DataFrame) -> "DeterministicPreprocessor":
        """Fits imputer medians, outlier fences, and standard scaler on training data ONLY."""
        self.numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
        
        # Calculate medians for imputation
        for col in self.numeric_cols:
            self.medians[col] = float(X[col].median()) if not X[col].dropna().empty else 0.0
            if self.clip_outliers:
                self.fences[col] = compute_iqr_fences(X[col])

        # Impute temporary copy to fit scaler
        X_imputed = X.copy()
        for col, med in self.medians.items():
            if col in X_imputed.columns:
                X_imputed[col] = X_imputed[col].fillna(med)

        if self.clip_outliers and self.fences:
            X_imputed = apply_fences(X_imputed, self.fences)

        if self.numeric_cols:
            self.scaler.fit(X_imputed[self.numeric_cols])

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms data using fitted training parameters."""
        X_out = X.copy()
        
        # Impute
        for col, med in self.medians.items():
            if col in X_out.columns:
                X_out[col] = X_out[col].fillna(med)

        # Winsorize
        if self.clip_outliers and self.fences:
            X_out = apply_fences(X_out, self.fences)

        # Scale
        if self.numeric_cols:
            valid_num = [c for c in self.numeric_cols if c in X_out.columns]
            if valid_num:
                X_out[valid_num] = self.scaler.transform(X_out[valid_num])

        return X_out

    def fit_transform(self, df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Convenience method: drops duplicates, fits on X, and transforms X."""
        cleaned = dedupe_with_log(df)
        X = cleaned.drop(columns=[target_col], errors="ignore")
        y = cleaned[target_col]
        
        self.fit(X)
        X_transformed = self.transform(X)
        return X_transformed, y
