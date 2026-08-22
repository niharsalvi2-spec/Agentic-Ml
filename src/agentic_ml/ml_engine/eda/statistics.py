"""
EDA Statistics and Profiling Engine.
Implements Freedman-Diaconis optimal bin estimation, skewness/kurtosis analysis,
and comprehensive multi-attribute statistical profiling.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import stats


def recommend_bins(data: pd.Series) -> int:
    """Freedman-Diaconis bin count. Robust to outliers (uses IQR rather than std)."""
    clean_data = data.dropna().to_numpy()
    n = len(clean_data)
    if n < 2:
        return 1
    q75, q25 = np.percentile(clean_data, [75, 25])
    iqr = q75 - q25
    if iqr == 0:
        return max(1, int(np.sqrt(n)))
    width = 2 * iqr / (n ** (1 / 3))
    if width == 0:
        return max(1, int(np.sqrt(n)))
    bins = int(np.ceil((clean_data.max() - clean_data.min()) / width))
    return max(1, min(bins, 100))


def skew_kurtosis(data: pd.Series) -> Dict[str, float]:
    """Calculates skewness and excess kurtosis."""
    clean_data = data.dropna()
    if len(clean_data) < 3:
        return {"skewness": 0.0, "kurtosis_excess": 0.0}
    return {
        "skewness": round(float(stats.skew(clean_data)), 4),
        "kurtosis_excess": round(float(stats.kurtosis(clean_data)), 4),
    }


def skew_label(skew_val: float) -> str:
    """Classifies skewness severity and provides modeling guidance."""
    if skew_val > 1.0:
        return "highly right-skewed (long tail high) — consider log/box-cox transform, use median imputation"
    elif skew_val > 0.5:
        return "moderately right-skewed — consider square root or log transform"
    elif skew_val < -1.0:
        return "highly left-skewed (long tail low) — consider reflect+log transform, use median imputation"
    elif skew_val < -0.5:
        return "moderately left-skewed"
    return "approximately symmetric — standard scaling and mean imputation are appropriate"


class EDAEngine:
    """Computes descriptive and correlation statistics for ML pipelines."""

    @staticmethod
    def analyze(df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
        numeric_df = df.select_dtypes(include=["number"])
        cat_df = df.select_dtypes(include=["object", "category"])
        
        skew_map = {}
        bins_map = {}
        for col in numeric_df.columns:
            sk = skew_kurtosis(numeric_df[col])
            skew_map[col] = {
                **sk,
                "shape": skew_label(sk["skewness"]),
                "recommended_bins": recommend_bins(numeric_df[col])
            }

        cat_summary = {}
        for col in cat_df.columns:
            counts = cat_df[col].value_counts()
            total = len(cat_df[col].dropna())
            top_share = float(counts.iloc[0] / total) if total > 0 and len(counts) > 0 else 0.0
            cat_summary[col] = {
                "cardinality": int(cat_df[col].nunique()),
                "top_category": str(counts.index[0]) if len(counts) > 0 else None,
                "top_share": round(top_share, 4),
                "is_imbalanced": bool(top_share > 0.6)
            }

        return {
            "n_rows": int(df.shape[0]),
            "n_columns": int(df.shape[1]),
            "summary_stats": numeric_df.describe().to_dict(),
            "skewness_and_kurtosis": skew_map,
            "categorical_distribution": cat_summary,
            "correlations": numeric_df.corr().to_dict() if numeric_df.shape[1] >= 2 else {},
        }
