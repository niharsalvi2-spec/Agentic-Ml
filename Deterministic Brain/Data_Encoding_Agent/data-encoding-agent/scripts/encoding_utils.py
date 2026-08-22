"""
encoding_utils.py
Reusable, leakage-safe building blocks for categorical encoding.
See ../references/code-generation.md for full usage patterns and rationale.
"""
import logging

import pandas as pd
from sklearn.model_selection import KFold

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("encoder")


def classify_cardinality(series: pd.Series) -> str:
    """Quick heuristic bucket: low / medium / high / very_high."""
    n = series.nunique()
    if n <= 10:
        return "low"
    elif n <= 50:
        return "medium"
    elif n <= 1000:
        return "high"
    return "very_high"


def fit_kfold_target_encoder(train_df: pd.DataFrame, cat_col: str, target_col: str,
                              n_splits: int = 5, smoothing: float = 100, seed: int = 42):
    """
    Leakage-safe target encoding. Returns (encoded_train_series, final_map, global_mean).
    final_map/global_mean should be applied to test/inference data via apply_target_encoder.
    """
    global_mean = train_df[target_col].mean()
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


def apply_target_encoder(df: pd.DataFrame, cat_col: str, final_map: pd.Series, global_mean: float) -> pd.Series:
    return df[cat_col].map(final_map).fillna(global_mean)


def leave_one_out_encode(df: pd.DataFrame, cat_col: str, target_col: str) -> pd.Series:
    sums = df.groupby(cat_col)[target_col].transform("sum")
    counts = df.groupby(cat_col)[target_col].transform("count")
    return (sums - df[target_col]) / (counts - 1)


def fit_frequency_encoder(train_df: pd.DataFrame, cat_col: str) -> pd.Series:
    return train_df[cat_col].value_counts(normalize=True)


def apply_frequency_encoder(df: pd.DataFrame, cat_col: str, freq_map: pd.Series) -> pd.Series:
    return df[cat_col].map(freq_map).fillna(0)


def encoding_report(original_df: pd.DataFrame, encoded_df: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
    """Cardinality vs. columns-produced audit — catches accidental dimensionality blowups."""
    rows = []
    for col in cat_cols:
        rows.append({
            "column": col,
            "cardinality": original_df[col].nunique(),
            "cardinality_bucket": classify_cardinality(original_df[col]),
            "columns_after_encoding": sum(1 for c in encoded_df.columns if str(c).startswith(col)),
        })
    report = pd.DataFrame(rows)
    logger.info(f"--- Encoding report ---\n{report.to_string(index=False)}")
    return report


def align_test_columns(train_encoded: pd.DataFrame, test_encoded: pd.DataFrame) -> pd.DataFrame:
    """After pd.get_dummies on train/test separately, align test columns to train's (fills 0 for missing)."""
    return test_encoded.reindex(columns=train_encoded.columns, fill_value=0)
