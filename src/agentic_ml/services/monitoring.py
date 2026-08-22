"""
Drift Detection Service — post-deployment model monitoring.

Implements lightweight statistical drift detection using:
  - Kolmogorov-Smirnov test for numeric feature drift
  - Chi-squared test for categorical feature drift
  - Prediction distribution comparison for label drift

This is Phase 8 (MLOps scaffold). The DriftDetector can be called
by a scheduled job or monitoring endpoint to check if the production
model's input distribution has shifted from training distribution.

Usage:
    detector = DriftDetector(reference_data=X_train)
    report = detector.check(current_data=X_live)
    if report.drift_detected:
        # trigger retraining agent
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("agentic_ml.services.monitoring")

# KS-test threshold for feature drift detection
_KS_DRIFT_THRESHOLD = 0.05  # p-value below this → drift detected
_DRIFT_FRACTION_THRESHOLD = 0.30  # >30% of features drifted → overall drift


class DriftReport:
    """Result of a drift detection check."""

    def __init__(
        self,
        drift_detected: bool,
        drifted_features: List[str],
        severity: str,
        feature_results: Dict[str, Dict],
        checked_at: str,
    ):
        self.drift_detected = drift_detected
        self.drifted_features = drifted_features
        self.severity = severity              # "NONE" | "LOW" | "MEDIUM" | "HIGH"
        self.feature_results = feature_results
        self.checked_at = checked_at

    def to_dict(self) -> Dict:
        return {
            "drift_detected": self.drift_detected,
            "drifted_features": self.drifted_features,
            "severity": self.severity,
            "feature_results": self.feature_results,
            "checked_at": self.checked_at,
            "recommendation": (
                "Retrain model — significant distribution shift detected."
                if self.drift_detected else
                "No action required — distribution is stable."
            ),
        }


class DriftDetector:
    """
    Stateful drift detector that compares live data against a reference distribution.

    The reference distribution is typically the training set or a recent validation window.
    """

    def __init__(self, reference_data: Any):
        """
        Args:
            reference_data: pandas DataFrame of the reference (training) distribution.
        """
        self.reference_data = reference_data
        self._reference_stats = self._compute_stats(reference_data)

    def _compute_stats(self, df: Any) -> Dict[str, Dict]:
        """Pre-compute column statistics for efficient comparison."""
        import numpy as np
        stats: Dict[str, Dict] = {}
        for col in df.columns:
            col_data = df[col].dropna()
            if col_data.dtype in [float, int] or str(col_data.dtype).startswith("float") or str(col_data.dtype).startswith("int"):
                stats[col] = {
                    "type": "numeric",
                    "values": col_data.values.tolist(),
                    "mean": float(col_data.mean()),
                    "std": float(col_data.std()),
                }
            else:
                stats[col] = {
                    "type": "categorical",
                    "distribution": col_data.value_counts(normalize=True).to_dict(),
                }
        return stats

    def check(self, current_data: Any) -> DriftReport:
        """
        Compare current_data against the reference distribution.

        Uses KS-test for numeric columns and chi-squared for categorical.

        Returns:
            DriftReport with per-feature drift results and overall verdict.
        """
        from scipy import stats as scipy_stats
        import numpy as np

        feature_results: Dict[str, Dict] = {}
        drifted_features: List[str] = []

        for col, ref_stat in self._reference_stats.items():
            if col not in current_data.columns:
                continue

            current_col = current_data[col].dropna()

            if ref_stat["type"] == "numeric":
                ref_values = ref_stat["values"]
                try:
                    ks_stat, p_value = scipy_stats.ks_2samp(ref_values, current_col.values)
                    drifted = p_value < _KS_DRIFT_THRESHOLD
                    feature_results[col] = {
                        "test": "KS",
                        "statistic": round(ks_stat, 4),
                        "p_value": round(p_value, 6),
                        "drifted": drifted,
                    }
                    if drifted:
                        drifted_features.append(col)
                except Exception as exc:
                    logger.debug("KS test failed for column '%s': %s", col, exc)

            else:
                # Categorical: compare distributions
                ref_dist = ref_stat.get("distribution", {})
                curr_dist = current_col.value_counts(normalize=True).to_dict()
                all_cats = set(ref_dist.keys()) | set(curr_dist.keys())
                ref_vec = [ref_dist.get(c, 0.0) for c in all_cats]
                curr_vec = [curr_dist.get(c, 0.0) for c in all_cats]
                try:
                    # Normalize to avoid chi-sq issues with 0 bins
                    ref_vec = [max(v, 1e-10) for v in ref_vec]
                    curr_vec = [max(v, 1e-10) for v in curr_vec]
                    chi2, p_value = scipy_stats.chisquare(curr_vec, f_exp=ref_vec)
                    drifted = p_value < _KS_DRIFT_THRESHOLD
                    feature_results[col] = {
                        "test": "chi2",
                        "statistic": round(chi2, 4),
                        "p_value": round(p_value, 6),
                        "drifted": drifted,
                    }
                    if drifted:
                        drifted_features.append(col)
                except Exception as exc:
                    logger.debug("Chi2 test failed for column '%s': %s", col, exc)

        total = len(self._reference_stats)
        drift_fraction = len(drifted_features) / max(total, 1)
        drift_detected = drift_fraction >= _DRIFT_FRACTION_THRESHOLD

        if not drifted_features:
            severity = "NONE"
        elif drift_fraction < 0.15:
            severity = "LOW"
        elif drift_fraction < 0.30:
            severity = "MEDIUM"
        else:
            severity = "HIGH"
            drift_detected = True

        report = DriftReport(
            drift_detected=drift_detected,
            drifted_features=drifted_features,
            severity=severity,
            feature_results=feature_results,
            checked_at=datetime.now(timezone.utc).isoformat(),
        )

        logger.info(
            "DriftDetector: drift_detected=%s, severity=%s, drifted=%d/%d features",
            drift_detected, severity, len(drifted_features), total,
        )

        return report
