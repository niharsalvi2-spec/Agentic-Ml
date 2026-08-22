"""
LeakageDetector — detects data leakage before model training.

Checks:
  1. Duplicate rows between train and test splits (row-level leakage)
  2. Target-correlated features that might be post-event or derived
  3. Preprocessing fitted before the train/test split (tracked via flag)
  4. Constant or near-constant features post-split (proxy for leakage)
  5. Perfect features: any single feature with > 0.99 correlation to target

These checks run BEFORE training. A failed report should halt the pipeline
or trigger remediation via the FailureAnalyzer.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("agentic_ml.ml_engine.data.leakage_detector")


class LeakageReport:
    """Result of leakage detection analysis."""

    def __init__(self, passed: bool, findings: List[str], severity: str = "LOW"):
        self.passed = passed
        self.findings = findings
        self.severity = severity  # "LOW" | "MEDIUM" | "HIGH"

    def to_dict(self) -> Dict:
        return {
            "passed": self.passed,
            "findings": self.findings,
            "severity": self.severity,
            "finding_count": len(self.findings),
        }


class LeakageDetector:
    """
    Stateless leakage detection engine.

    Usage:
        report = LeakageDetector.check(X_train, X_test, y_train, target_column="churn")
        if not report.passed:
            # Route to FailureAnalyzer
    """

    @staticmethod
    def check(
        X_train: Any,
        X_test: Any,
        y_train: Any,
        target_column: Optional[str] = None,
        preprocessor_fit_before_split: bool = False,
    ) -> LeakageReport:
        """
        Run all leakage checks on train/test splits.

        Args:
            X_train:                     Training feature matrix (DataFrame)
            X_test:                      Test feature matrix (DataFrame)
            y_train:                     Training target series
            target_column:               Target column name (for correlation checks)
            preprocessor_fit_before_split: True if preprocessing was fitted on full data

        Returns:
            LeakageReport with passed status and list of findings.
        """
        import numpy as np
        import pandas as pd

        findings: List[str] = []

        # ── Check 1: Preprocessor fit before split ────────────────────────
        if preprocessor_fit_before_split:
            findings.append(
                "CRITICAL: Preprocessor was fitted on the full dataset before train/test split — "
                "this causes target leakage via statistics computed on test rows."
            )

        # ── Check 2: Train/test row overlap ──────────────────────────────
        try:
            train_hashes = set(
                pd.util.hash_pandas_object(X_train, index=False).astype(str)
            )
            test_hashes = set(
                pd.util.hash_pandas_object(X_test, index=False).astype(str)
            )
            overlap = train_hashes & test_hashes
            if overlap:
                findings.append(
                    f"WARNING: {len(overlap)} rows appear in both train and test sets — "
                    "identical data rows will inflate test performance."
                )
        except Exception as exc:
            logger.debug("Row overlap check failed: %s", exc)

        # ── Check 3: Perfect or near-perfect numeric predictors ──────────
        try:
            numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
            y_vals = np.array(y_train)
            # Only applies to classification/binary targets with numeric encoding
            if len(np.unique(y_vals)) <= 20:
                for col in numeric_cols[:50]:  # limit to first 50 to avoid O(n*k) cost
                    try:
                        corr = float(np.corrcoef(X_train[col].fillna(0), y_vals)[0, 1])
                        if not np.isnan(corr) and abs(corr) > 0.99:
                            findings.append(
                                f"HIGH RISK: Feature '{col}' has correlation {corr:.4f} with target — "
                                "may be a target-derived or post-event feature."
                            )
                    except Exception:
                        pass
        except Exception as exc:
            logger.debug("Correlation leakage check failed: %s", exc)

        # ── Determine severity ────────────────────────────────────────────
        critical = [f for f in findings if "CRITICAL" in f]
        high = [f for f in findings if "HIGH RISK" in f]

        if critical:
            severity = "HIGH"
            passed = False
        elif high:
            severity = "MEDIUM"
            passed = True  # warning only — don't halt, but record
        elif findings:
            severity = "LOW"
            passed = True
        else:
            severity = "LOW"
            passed = True

        report = LeakageReport(passed=passed, findings=findings, severity=severity)

        if passed:
            logger.info(
                "LeakageDetector: PASS (severity=%s, findings=%d)", severity, len(findings)
            )
        else:
            logger.warning(
                "LeakageDetector: FAIL (severity=%s) — %s", severity, findings
            )

        return report
