"""
Outlier and Anomaly Detection module.
Implements Tukey IQR fences, Modified Z-score (MAD), and Isolation Forest detection.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from src.agentic_ml.ml_engine.preprocessing.cleaner import compute_iqr_fences, modified_z_outliers


class OutlierDetector:
    """Detects point outliers using statistical and isolation methods."""

    @staticmethod
    def detect_iqr_outliers(series: pd.Series, k: float = 1.5) -> Dict[str, Any]:
        lower, upper = compute_iqr_fences(series, k=k)
        mask = (series < lower) | (series > upper)
        return {
            "lower_fence": lower,
            "upper_fence": upper,
            "outlier_indices": series[mask].index.tolist(),
            "outlier_count": int(mask.sum()),
        }

    @staticmethod
    def detect_mad_outliers(series: pd.Series, threshold: float = 3.5) -> Dict[str, Any]:
        mask = modified_z_outliers(series, threshold=threshold)
        return {
            "threshold": threshold,
            "outlier_indices": series[mask].index.tolist(),
            "outlier_count": int(mask.sum()),
        }
