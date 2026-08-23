"""
ModelRiskScorer — Deterministic, explainable deployment risk assessment for ML models.

Risk scoring model:
  - Heuristic risk estimation computed across 5 transparent risk dimensions.
  - Direction-aware primary metric evaluation.
  - Clamped score: 0–100.
  - Governed by DeploymentPolicy thresholds:
      LOW:    0–39  -> AUTO_APPROVE (requires_hitl = False)
      MEDIUM: 40–69 -> HUMAN_REQUIRED (requires_hitl = True)
      HIGH:   70–100 -> HUMAN_REQUIRED (requires_hitl = True)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from src.agentic_ml.governance.policy import DEFAULT_DEPLOYMENT_POLICY, DeploymentPolicy
from src.agentic_ml.ml_engine.evaluation.metrics import (
    PrimaryMetric,
    extract_primary_metric,
)

logger = logging.getLogger("agentic_ml.ml_engine.evaluation.risk_scorer")


class RiskScore:
    """Result of risk scoring."""

    def __init__(
        self,
        score: int,
        risk_level: str,
        deployment_decision: str,
        factors: Dict[str, Any],
        requires_hitl: bool,
    ):
        self.score = score
        self.risk_level = risk_level
        self.deployment_decision = deployment_decision   # "AUTO_APPROVE" | "HUMAN_REQUIRED"
        self.factors = factors
        self.requires_hitl = requires_hitl

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "risk_level": self.risk_level,
            "deployment_decision": self.deployment_decision,
            "factors": self.factors,
            "requires_hitl": self.requires_hitl,
        }


class ModelRiskScorer:
    """
    Computes a deterministic, explainable deployment risk score from validation metrics and dataset profile.
    """

    @staticmethod
    def score(
        metrics: Dict[str, float],
        task_type: str = "classification",
        dataset_profile: Optional[Dict[str, Any]] = None,
        train_score: Optional[float] = None,
        cv_std: Optional[float] = None,
        primary_metric: Optional[PrimaryMetric] = None,
        policy: Optional[DeploymentPolicy] = None,
    ) -> RiskScore:
        """
        Compute deployment risk score respecting metric direction, sample size, variance, and class imbalance.
        """
        active_policy = policy or DEFAULT_DEPLOYMENT_POLICY
        raw_score = 0
        factors: Dict[str, Any] = {}

        if primary_metric is None:
            primary_metric = extract_primary_metric(metrics, task_type=task_type)

        metric_name = primary_metric.name
        metric_val = primary_metric.value
        direction = primary_metric.direction

        factors["primary_metric"] = {
            "name": metric_name,
            "value": metric_val,
            "direction": direction,
        }

        # ── Factor 1: Performance level (Direction-Aware) ────────────────
        if direction == "maximize":
            # Higher is better (Accuracy, F1, ROC-AUC, R2, Silhouette)
            if metric_val < 0.55:
                raw_score += 35
                factors["performance_risk"] = f"HIGH — {metric_name}={metric_val:.3f} (<0.55)"
            elif metric_val < 0.70:
                raw_score += 20
                factors["performance_risk"] = f"MEDIUM — {metric_name}={metric_val:.3f} (<0.70)"
            elif metric_val < 0.85:
                raw_score += 8
                factors["performance_risk"] = f"LOW — {metric_name}={metric_val:.3f} (<0.85)"
            else:
                factors["performance_risk"] = "NONE — strong benchmark performance"
        else:
            # Lower is better (RMSE, MAE, MSE, Loss)
            if metric_val > 1.5:
                raw_score += 35
                factors["performance_risk"] = f"HIGH — error metric {metric_name}={metric_val:.3f} (>1.5)"
            elif metric_val > 0.8:
                raw_score += 20
                factors["performance_risk"] = f"MEDIUM — error metric {metric_name}={metric_val:.3f} (>0.8)"
            else:
                factors["performance_risk"] = f"LOW — acceptable error {metric_name}={metric_val:.3f}"

        # ── Factor 2: Dataset size ────────────────────────────────────────
        profile = dataset_profile or {}
        row_count = profile.get("row_count") or profile.get("n_rows") or profile.get("samples") or 0
        factors["row_count"] = row_count

        if row_count > 0:
            if row_count < 100:
                raw_score += 30
                factors["data_size_risk"] = f"HIGH — critical small sample size ({row_count} rows)"
            elif row_count < 500:
                raw_score += 15
                factors["data_size_risk"] = f"MEDIUM — modest sample size ({row_count} rows)"
            else:
                factors["data_size_risk"] = f"LOW — adequate sample size ({row_count} rows)"

        # ── Factor 3: Overfitting gap ─────────────────────────────────────
        if train_score is not None and direction == "maximize":
            gap = max(0.0, train_score - metric_val)
            factors["train_cv_gap"] = round(gap, 4)
            if gap > 0.20:
                raw_score += 20
                factors["overfitting_risk"] = f"HIGH — train/cv gap={gap:.3f}"
            elif gap > 0.10:
                raw_score += 10
                factors["overfitting_risk"] = f"MEDIUM — train/cv gap={gap:.3f}"
            else:
                factors["overfitting_risk"] = "LOW — negligible overfitting"

        # ── Factor 4: CV instability ───────────────────────────────────────
        if cv_std is not None:
            factors["cv_std"] = round(cv_std, 4)
            if cv_std > 0.12:
                raw_score += 15
                factors["stability_risk"] = f"HIGH — cross-validation std={cv_std:.3f}"
            elif cv_std > 0.06:
                raw_score += 8
                factors["stability_risk"] = f"MEDIUM — cross-validation std={cv_std:.3f}"
            else:
                factors["stability_risk"] = "LOW — stable cross-validation"

        # ── Factor 5: Real Class imbalance ────────────────────────────────
        minority_pct = profile.get("minority_pct")
        if minority_pct is None and "class_distribution" in profile:
            dist = profile["class_distribution"]
            if isinstance(dist, dict) and dist:
                minority_pct = min(dist.values()) * 100 if all(isinstance(v, (int, float)) for v in dist.values()) else None

        if minority_pct is not None and task_type == "classification":
            factors["minority_percentage"] = round(float(minority_pct), 2)
            if minority_pct < 10.0:
                raw_score += 20
                factors["imbalance_risk"] = f"HIGH — severe minority class imbalance ({minority_pct:.1f}%)"
            elif minority_pct < 20.0:
                raw_score += 10
                factors["imbalance_risk"] = f"MEDIUM — moderate class imbalance ({minority_pct:.1f}%)"
            else:
                factors["imbalance_risk"] = "LOW — balanced class distribution"

        # Clamp to [0, 100]
        final_score = min(100, max(0, raw_score))
        factors["raw_score"] = raw_score
        factors["clamped_score"] = final_score

        # ── Determine risk level and decision from central policy ─────────────
        risk_level, decision, requires_hitl = active_policy.evaluate(final_score)

        logger.info(
            "ModelRiskScorer: score=%d, risk=%s, decision=%s, requires_hitl=%s",
            final_score, risk_level, decision, requires_hitl
        )

        return RiskScore(
            score=final_score,
            risk_level=risk_level,
            deployment_decision=decision,
            factors=factors,
            requires_hitl=requires_hitl,
        )
