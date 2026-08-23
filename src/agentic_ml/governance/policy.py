"""
DeploymentPolicy — Central authority defining deployment risk thresholds and HITL governance rules.

Policy Contract:
  - LOW risk (0 <= score <= 39): AUTO_APPROVE (requires_hitl = False)
  - MEDIUM risk (40 <= score <= 69): HUMAN_REQUIRED (requires_hitl = True)
  - HIGH risk (70 <= score <= 100): HUMAN_REQUIRED (requires_hitl = True)

No silent conversion of HUMAN_REVIEW/MEDIUM into AUTO_APPROVE.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class DeploymentPolicy:
    auto_approve_max_score: int = 39      # Score 0..39 -> AUTO_APPROVE
    medium_risk_max_score: int = 69       # Score 40..69 -> MEDIUM (HUMAN_REQUIRED)
    # Score 70..100 -> HIGH (HUMAN_REQUIRED)

    def evaluate(self, risk_score: int) -> Tuple[str, str, bool]:
        """
        Evaluate a risk score and return (risk_level, deployment_decision, requires_hitl).

        Returns:
            Tuple of:
              - risk_level: "LOW" | "MEDIUM" | "HIGH"
              - deployment_decision: "AUTO_APPROVE" | "HUMAN_REQUIRED"
              - requires_hitl: bool
        """
        if not isinstance(risk_score, (int, float)):
            raise TypeError(f"Risk score must be numeric; got {type(risk_score).__name__}")

        clamped = max(0, min(100, int(round(risk_score))))

        if clamped <= self.auto_approve_max_score:
            return "LOW", "AUTO_APPROVE", False
        elif clamped <= self.medium_risk_max_score:
            return "MEDIUM", "HUMAN_REQUIRED", True
        else:
            return "HIGH", "HUMAN_REQUIRED", True


# Default singleton policy
DEFAULT_DEPLOYMENT_POLICY = DeploymentPolicy()


def evaluate_deployment_governance(risk_score: int) -> Tuple[str, str, bool]:
    """Helper to evaluate risk score against active deployment policy."""
    return DEFAULT_DEPLOYMENT_POLICY.evaluate(risk_score)
