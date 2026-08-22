"""
cleaning_utils.py
Reusable, leakage-safe building blocks for data-cleaning scripts.
See ../references/code-generation.md for full usage patterns and rationale.
"""
import logging

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("cleaner")


def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column missing count/percentage — always run this first."""
    report = pd.DataFrame({
        "missing_count": df.isnull().sum(),
        "missing_pct": (df.isnull().sum() / len(df) * 100).round(2),
        "dtype": df.dtypes,
    }).sort_values("missing_pct", ascending=False)
    return report


def compute_iqr_fences(series: pd.Series, k: float = 1.5):
    """Compute Tukey IQR fences on a (train-only) series. Returns (lower, upper)."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def apply_fences(df: pd.DataFrame, fences: dict) -> pd.DataFrame:
    """Clip (Winsorize) columns to pre-computed fences. fences: {col: (lower, upper)}."""
    df = df.copy()
    for col, (lower, upper) in fences.items():
        df[col] = df[col].clip(lower, upper)
    return df


def modified_z_outliers(series: pd.Series, threshold: float = 3.5) -> pd.Series:
    """Boolean mask of outliers via MAD-based modified Z-score (robust to skew)."""
    median = series.median()
    mad = (series - median).abs().median()
    if mad == 0:
        return pd.Series(False, index=series.index)
    modified_z = 0.6745 * (series - median) / mad
    return modified_z.abs() > threshold


def dedupe_with_log(df: pd.DataFrame, subset=None, keep: str = "first") -> pd.DataFrame:
    """Drop duplicates, preferring the most complete row per key, with a logged count."""
    before_n = len(df)
    df = df.copy()
    df["_completeness"] = df.notnull().sum(axis=1)
    if subset:
        df = (df.sort_values("_completeness", ascending=False)
                .drop_duplicates(subset=subset, keep="first"))
    else:
        df = df.drop_duplicates(keep=keep)
    df = df.drop(columns="_completeness")
    logger.info(f"Dedup: {before_n} -> {len(df)} rows ({before_n - len(df)} removed)")
    return df


def cleaning_report(before: pd.DataFrame, after: pd.DataFrame, label: str = "") -> pd.DataFrame:
    """Before/after row count + missingness diff for a cleaning step."""
    rows = []
    for col in before.columns:
        if col not in after.columns:
            continue
        rows.append({
            "column": col,
            "missing_before": before[col].isnull().sum(),
            "missing_after": after[col].isnull().sum(),
        })
    report = pd.DataFrame(rows)
    if label:
        logger.info(f"--- Cleaning report: {label} ({len(before)} -> {len(after)} rows) ---")
    return report


def classify_missingness_hint(df: pd.DataFrame, target_col: str, other_cols=None, alpha=0.05):
    """
    Rough MCAR-vs-not signal: t-test comparing `other_cols` between rows where target_col
    is missing vs not. A significant difference suggests MAR/MNAR rather than MCAR.
    This is a heuristic, not a proof — always combine with domain reasoning.
    """
    from scipy import stats

    other_cols = other_cols or df.select_dtypes(include="number").columns.drop(
        target_col, errors="ignore"
    )
    is_missing = df[target_col].isnull()
    results = {}
    for col in other_cols:
        if col == target_col:
            continue
        a = df.loc[is_missing, col].dropna()
        b = df.loc[~is_missing, col].dropna()
        if len(a) < 2 or len(b) < 2:
            continue
        _, p = stats.ttest_ind(a, b, equal_var=False)
        results[col] = {"p_value": round(p, 4), "likely_not_MCAR": p < alpha}
    return pd.DataFrame(results).T
