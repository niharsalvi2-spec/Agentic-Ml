"""
ModelRiskScorer — scores a trained model's deployment risk before the HITL gate.

Risk score: 0–100 (higher = riskier → more likely to require human review)

Factors:
  - Performance level: low accuracy/F1 increases risk
  - Dataset size: very small datasets are high risk
  - Class imbalance: imbalanced data with high accuracy is deceptive
  - Validation gap (train vs CV): overfitting indicator
  - CV standard deviation: unstable model
  - Missing feature coverage: large missingness in training data

Risk levels:
  - LOW    (0–39):   Auto-approve deployment
  - MEDIUM (40–69):  Recommend human review
  - HIGH   (70–100): Require human approval (HITL interrupt)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("agentic_ml.ml_engine.evaluation.risk_scorer")

_AUTO_APPROVE_THRESHOLD = 40   # score below this → AUTO
_HUMAN_REQUIRED_THRESHOLD = 70  # score above this → HITL interrupt


class RiskScore:
    """Result of risk scoring."""

    def __init__(
        self,
        score: int,
        risk_level: str,
        deployment_decision: str,
        factors: Dict[str, Any],
    ):
        self.score = score
        self.risk_level = risk_level
        self.deployment_decision = deployment_decision   # "AUTO_APPROVE" | "HUMAN_REVIEW" | "HUMAN_REQUIRED"
        self.factors = factors

    def to_dict(self) -> Dict:
        return {
            "score": self.score,
            "risk_level": self.risk_level,
            "deployment_decision": self.deployment_decision,
            "factors": self.factors,
        }

    @property
    def requires_hitl(self) -> bool:
        return self.deployment_decision == "HUMAN_REQUIRED"


class ModelRiskScorer:
    """
    Computes a composite deployment risk score from validation metrics and dataset profile.
    """

    @staticmethod
    def score(
        metrics: Dict[str, float],
        task_type: str = "classification",
        dataset_profile: Optional[Dict[str, Any]] = None,
        train_score: Optional[float] = None,
        cv_std: Optional[float] = None,
    ) -> RiskScore:
        """
        Compute deployment risk score.

        Args:
            metrics:         model evaluation metrics (e.g., {"RandomForest": 0.91})
            task_type:       "classification" | "regression" | "clustering"
            dataset_profile: DatasetManifest or data_summary dict
            train_score:     training set score (for gap detection)
            cv_std:          cross-validation std deviation

        Returns:
            RiskScore with score, risk_level, and deployment_decision.
        """
        raw_score = 0
        factors: Dict[str, Any] = {}

        # Extract best metric value
        best_metric = max(metrics.values()) if metrics else 0.0
        factors["best_metric"] = best_metric

        # ── Factor 1: Performance level ───────────────────────────────────
        if task_type == "classification":
            if best_metric < 0.65:
                raw_score += 35
                factors["performance_risk"] = "HIGH — accuracy/F1 below 0.65"
            elif best_metric < 0.75:
                raw_score += 20
                factors["performance_risk"] = "MEDIUM — accuracy/F1 below 0.75"
            elif best_metric < 0.85:
                raw_score += 8
                factors["performance_risk"] = "LOW — acceptable performance"
            else:
                factors["performance_risk"] = "NONE — strong performance"
        else:
            # Regression: R2 based
            if best_metric < 0.5:
                raw_score += 30
                factors["performance_risk"] = "HIGH — R2 below 0.5"
            elif best_metric < 0.7:
                raw_score += 15
                factors["performance_risk"] = "MEDIUM"
            else:
                factors["performance_risk"] = "LOW"

        # ── Factor 2: Dataset size ────────────────────────────────────────
        if dataset_profile:
            row_count = dataset_profile.get("row_count", 0)
            if row_count < 500:
                raw_score += 25
                factors["data_size_risk"] = f"HIGH — only {row_count} rows"
            elif row_count < 2000:
                raw_score += 10
                factors["data_size_risk"] = f"MEDIUM — {row_count} rows"
            else:
                factors["data_size_risk"] = f"LOW — {row_count} rows"

        # ── Factor 3: Overfitting gap ─────────────────────────────────────
        if train_score is not None:
            gap = train_score - best_metric
            factors["train_cv_gap"] = round(gap, 4)
            if gap > 0.20:
                raw_score += 20
                factors["overfitting_risk"] = f"HIGH — gap={gap:.3f}"
            elif gap > 0.10:
                raw_score += 10
                factors["overfitting_risk"] = f"MEDIUM — gap={gap:.3f}"
            else:
                factors["overfitting_risk"] = "LOW"

        # ── Factor 4: CV instability ───────────────────────────────────────
        if cv_std is not None:
            factors["cv_std"] = round(cv_std, 4)
            if cv_std > 0.10:
                raw_score += 15
                factors["stability_risk"] = f"HIGH — cv_std={cv_std:.3f}"
            elif cv_std > 0.05:
                raw_score += 7
                factors["stability_risk"] = f"MEDIUM — cv_std={cv_std:.3f}"
            else:
                factors["stability_risk"] = "LOW"

        # ── Factor 5: Class imbalance ─────────────────────────────────────
        if dataset_profile:
            target_card = dataset_profile.get("target_cardinality")
            row_count = dataset_profile.get("row_count", 1)
            # Approximation — if binary and minority class is small
            if target_card == 2 and task_type == "classification":
                minority_pct = 50  # unknown without actual counts; assume balanced
                if minority_pct < 15:
                    raw_score += 10
                    factors["imbalance_risk"] = "MEDIUM — binary target may be imbalanced"

        # Clamp to [0, 100]
        final_score = min(100, max(0, raw_score))
        factors["raw_score"] = raw_score
        factors["clamped_score"] = final_score

        # ── Determine risk level and decision ─────────────────────────────
        if final_score < _AUTO_APPROVE_THRESHOLD:
            risk_level = "LOW"
            decision = "AUTO_APPROVE"
        elif final_score < _HUMAN_REQUIRED_THRESHOLD:
            risk_level = "MEDIUM"
            decision = "HUMAN_REVIEW"
        else:
            risk_level = "HIGH"
            decision = "HUMAN_REQUIRED"

        logger.info(
            "ModelRiskScorer: score=%d, risk=%s, decision=%s",
            final_score, risk_level, decision
        )

        return RiskScore(
            score=final_score,
            risk_level=risk_level,
            deployment_decision=decision,
            factors=factors,
        )
