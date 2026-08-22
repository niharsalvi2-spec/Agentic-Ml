"""
Distribution analysis module for univariate features.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from src.agentic_ml.ml_engine.eda.statistics import recommend_bins, skew_kurtosis, skew_label


class DistributionAnalyzer:
    """Analyzes univariate feature distributions, bounds, and imbalance."""

    @staticmethod
    def analyze_numeric(series: pd.Series) -> Dict[str, Any]:
        clean = series.dropna()
        n = len(clean)
        if n == 0:
            return {"count": 0}

        bins = recommend_bins(clean)
        sk = skew_kurtosis(clean)
        q1, q3 = float(clean.quantile(0.25)), float(clean.quantile(0.75))
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = clean[(clean < lower) | (clean > upper)]

        return {
            "n": n,
            "mean": round(float(clean.mean()), 4),
            "std": round(float(clean.std()), 4),
            "median": round(float(clean.median()), 4),
            "iqr_fences": (round(lower, 4), round(upper, 4)),
            "outlier_count": int(len(outliers)),
            "outlier_pct": round(100.0 * len(outliers) / max(1, n), 2),
            "bins": bins,
            **sk,
            "shape": skew_label(sk["skewness"]),
        }

    @staticmethod
    def analyze_categorical(series: pd.Series, top_n: int = 10) -> Dict[str, Any]:
        clean = series.dropna()
        n = len(clean)
        if n == 0:
            return {"cardinality": 0}

        counts = clean.value_counts()
        top_share = float(counts.iloc[0] / n) if n > 0 else 0.0

        return {
            "cardinality": int(clean.nunique()),
            "top_categories": counts.head(top_n).to_dict(),
            "top_category_share": round(top_share, 4),
            "is_imbalanced": bool(top_share > 0.6),
        }
