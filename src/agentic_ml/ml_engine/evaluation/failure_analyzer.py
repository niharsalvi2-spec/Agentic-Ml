"""
FailureAnalyzer — diagnoses WHY validation failed and recommends remediation.

Called by the validation agent when cross-validation score falls below threshold
or when LeakageDetector reports a critical finding.

Failure categories:
  - poor_recall        → model misses too many positives (increase recall weight)
  - high_variance      → CV std deviation too high (overfitting)
  - overfitting        → large train/val gap
  - class_imbalance    → severe imbalance not handled
  - bad_preprocessing  → preprocessing artefacts (scaling issues, leakage)
  - data_leakage       → leakage detector critical finding
  - low_sample         → dataset too small for reliable CV
  - feature_instability → high std in feature importance across folds

Remediation actions map to LangGraph routing decisions.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("agentic_ml.ml_engine.evaluation.failure_analyzer")

# Thresholds
_MIN_ACCEPTABLE_SCORE = 0.55
_HIGH_VARIANCE_STD_THRESHOLD = 0.08
_OVERFITTING_GAP_THRESHOLD = 0.15
_MIN_RECALL_FOR_IMBALANCED = 0.60


class FailureAnalysis:
    """Structured result from failure analysis."""

    def __init__(
        self,
        failure_categories: List[str],
        root_cause: str,
        remediation_action: str,
        retry_allowed: bool,
        details: Dict[str, Any],
    ):
        self.failure_categories = failure_categories
        self.root_cause = root_cause
        self.remediation_action = remediation_action
        self.retry_allowed = retry_allowed
        self.details = details

    def to_dict(self) -> Dict:
        return {
            "failure_categories": self.failure_categories,
            "root_cause": self.root_cause,
            "remediation_action": self.remediation_action,
            "retry_allowed": self.retry_allowed,
            "details": self.details,
        }


class FailureAnalyzer:
    """
    Analyzes validation failure and recommends a remediation action.

    Remediation actions:
      - "retry_with_different_model"       → model_building with different candidates
      - "retry_with_resampling"            → model_building with SMOTE/class_weight
      - "retry_preprocessing"              → preprocessing needs adjustment
      - "flag_and_stop"                    → unrecoverable; halt pipeline with evidence
    """

    @staticmethod
    def analyze(
        best_score: float,
        all_scores: Dict[str, float],
        cv_std: Optional[float] = None,
        train_score: Optional[float] = None,
        task_type: str = "classification",
        class_balance: Optional[float] = None,   # minority class fraction
        leakage_report: Optional[Dict] = None,
    ) -> FailureAnalysis:
        """
        Diagnose why validation failed.

        Args:
            best_score:     best CV score achieved
            all_scores:     all model CV scores
            cv_std:         standard deviation of CV scores (if available)
            train_score:    training set score for overfitting detection
            task_type:      classification | regression | clustering
            class_balance:  fraction of minority class (e.g. 0.18 = 18% positive)
            leakage_report: LeakageReport.to_dict() result if available

        Returns:
            FailureAnalysis with root cause, action, and retry decision.
        """
        categories: List[str] = []
        details: Dict[str, Any] = {
            "best_score": best_score,
            "all_scores": all_scores,
            "cv_std": cv_std,
            "train_score": train_score,
            "class_balance": class_balance,
        }

        # ── Check for critical leakage ─────────────────────────────────
        if leakage_report and not leakage_report.get("passed", True):
            categories.append("data_leakage")
            details["leakage_findings"] = leakage_report.get("findings", [])

        # ── Score too low ─────────────────────────────────────────────
        if best_score < _MIN_ACCEPTABLE_SCORE:
            # Try to diagnose WHY

            # Class imbalance: minority class < 20% and score is accuracy-dominated
            if class_balance is not None and class_balance < 0.20:
                categories.append("class_imbalance")

            # Low sample size (heuristic: all models score within 5% of each other)
            score_range = max(all_scores.values()) - min(all_scores.values()) if all_scores else 0
            if score_range < 0.05 and best_score < 0.65:
                categories.append("low_sample")

        # ── High variance across CV folds ────────────────────────────
        if cv_std is not None and cv_std > _HIGH_VARIANCE_STD_THRESHOLD:
            categories.append("high_variance")

        # ── Overfitting: large train/val gap ─────────────────────────
        if train_score is not None and (train_score - best_score) > _OVERFITTING_GAP_THRESHOLD:
            categories.append("overfitting")

        # ── Poor recall (checked via metrics if available) ────────────
        if task_type == "classification" and class_balance is not None and class_balance < 0.30:
            if best_score < _MIN_RECALL_FOR_IMBALANCED:
                categories.append("poor_recall")

        # If nothing specific detected, mark as generic low performance
        if not categories:
            categories.append("low_performance")

        # ── Determine remediation action ──────────────────────────────
        if "data_leakage" in categories:
            root_cause = "Critical data leakage detected — model results are unreliable."
            action = "flag_and_stop"
            retry = False

        elif "class_imbalance" in categories or "poor_recall" in categories:
            root_cause = (
                f"Class imbalance (minority={class_balance:.1%}) causing poor recall. "
                f"Best score: {best_score:.4f}."
            )
            action = "retry_with_resampling"
            retry = True

        elif "overfitting" in categories or "high_variance" in categories:
            root_cause = (
                f"Model is overfitting (train={train_score:.4f}, cv={best_score:.4f}) "
                f"or unstable (cv_std={cv_std:.4f})."
            )
            action = "retry_with_different_model"
            retry = True

        elif "low_sample" in categories:
            root_cause = (
                f"Dataset appears too small for reliable CV — all models converge near {best_score:.4f}."
            )
            action = "retry_preprocessing"  # feature engineering may help
            retry = True

        else:
            root_cause = f"No model exceeded minimum acceptable score ({best_score:.4f} < {_MIN_ACCEPTABLE_SCORE})."
            action = "retry_with_different_model"
            retry = True

        logger.info(
            "FailureAnalyzer: categories=%s, action=%s, retry=%s", categories, action, retry
        )

        return FailureAnalysis(
            failure_categories=categories,
            root_cause=root_cause,
            remediation_action=action,
            retry_allowed=retry,
            details=details,
        )
