"""
Correlation and Multicollinearity Analyzer.
Identifies highly correlated feature pairs (|r| > threshold) and suggests drops.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


class CorrelationAnalyzer:
    """Detects linear correlations and multicollinearity in numerical feature matrices."""

    @staticmethod
    def analyze_correlations(
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        threshold: float = 0.80
    ) -> Dict[str, Any]:
        num_df = df.select_dtypes(include=[np.number])
        if num_df.shape[1] < 2:
            return {"correlation_matrix": {}, "multicollinear_pairs": []}

        corr = num_df.corr(method="pearson")
        cols = corr.columns.tolist()
        flagged_pairs = []

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                c1, c2 = cols[i], cols[j]
                r_val = float(corr.iloc[i, j])
                if abs(r_val) >= threshold:
                    pair_info: Dict[str, Any] = {
                        "feature_1": c1,
                        "feature_2": c2,
                        "pearson_r": round(r_val, 4)
                    }
                    if target_col and target_col in corr.columns:
                        t1 = abs(float(corr.loc[c1, target_col])) if c1 != target_col else 1.0
                        t2 = abs(float(corr.loc[c2, target_col])) if c2 != target_col else 1.0
                        pair_info["suggest_drop"] = c1 if t1 < t2 else c2
                    flagged_pairs.append(pair_info)

        return {
            "correlation_matrix": corr.round(4).to_dict(),
            "multicollinear_pairs": flagged_pairs,
            "threshold_used": threshold,
        }
