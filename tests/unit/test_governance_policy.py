"""
Tests for Centralized Deployment Governance Policy and Risk Boundaries (Phase 3 & 9).
"""
import pytest

from src.agentic_ml.governance.policy import DeploymentPolicy, DEFAULT_DEPLOYMENT_POLICY
from src.agentic_ml.ml_engine.evaluation.risk_scorer import ModelRiskScorer


class TestGovernancePolicy:

    def test_policy_boundaries_and_levels(self):
        policy = DeploymentPolicy()

        # Score 0 (boundary) -> LOW, AUTO_APPROVE, requires_hitl=False
        level, decision, hitl = policy.evaluate(0)
        assert (level, decision, hitl) == ("LOW", "AUTO_APPROVE", False)

        # Score 39 (boundary) -> LOW, AUTO_APPROVE, requires_hitl=False
        level, decision, hitl = policy.evaluate(39)
        assert (level, decision, hitl) == ("LOW", "AUTO_APPROVE", False)

        # Score 40 (boundary) -> MEDIUM, HUMAN_REQUIRED, requires_hitl=True
        level, decision, hitl = policy.evaluate(40)
        assert (level, decision, hitl) == ("MEDIUM", "HUMAN_REQUIRED", True)

        # Score 69 (boundary) -> MEDIUM, HUMAN_REQUIRED, requires_hitl=True
        level, decision, hitl = policy.evaluate(69)
        assert (level, decision, hitl) == ("MEDIUM", "HUMAN_REQUIRED", True)

        # Score 70 (boundary) -> HIGH, HUMAN_REQUIRED, requires_hitl=True
        level, decision, hitl = policy.evaluate(70)
        assert (level, decision, hitl) == ("HIGH", "HUMAN_REQUIRED", True)

        # Score 100 (boundary) -> HIGH, HUMAN_REQUIRED, requires_hitl=True
        level, decision, hitl = policy.evaluate(100)
        assert (level, decision, hitl) == ("HIGH", "HUMAN_REQUIRED", True)

    def test_malformed_input_rejection(self):
        policy = DeploymentPolicy()
        with pytest.raises(TypeError):
            policy.evaluate("invalid_string_score")  # type: ignore

    def test_risk_scorer_deterministic_scoring(self):
        # Perfect model on large dataset -> Low Risk Auto Approve
        risk_low = ModelRiskScorer.score(
            metrics={"f1": 0.95, "accuracy": 0.96},
            task_type="classification",
            dataset_profile={"row_count": 5000, "minority_pct": 45.0},
        )
        assert risk_low.score < 40
        assert risk_low.risk_level == "LOW"
        assert risk_low.deployment_decision == "AUTO_APPROVE"
        assert risk_low.requires_hitl is False

        # Fragile model on tiny dataset with high variance -> High Risk HITL Required
        risk_high = ModelRiskScorer.score(
            metrics={"f1": 0.52},
            task_type="classification",
            dataset_profile={"row_count": 50, "minority_pct": 5.0},
            cv_std=0.18,
            train_score=0.95,
        )
        assert risk_high.score >= 70
        assert risk_high.risk_level == "HIGH"
        assert risk_high.deployment_decision == "HUMAN_REQUIRED"
        assert risk_high.requires_hitl is True
