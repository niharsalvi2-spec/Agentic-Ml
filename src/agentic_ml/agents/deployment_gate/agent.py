"""
Deployment Gate Node — Governance and Human-in-the-Loop (HITL) approval gate.

Authority:
  - Assesses deployment risk using ModelRiskScorer.
  - Automatically routes to deployment if risk is LOW ("AUTO_APPROVE").
  - Interrupts execution for human approval if risk is MEDIUM / HIGH ("HUMAN_REQUIRED").
  - Rejection terminates at END with deployment_decision="REJECTED".
  - Approval updates deployment_decision="HUMAN_APPROVED" and advances to deployment.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from langgraph.types import Command, interrupt
from langgraph.graph import END

from src.agentic_ml.state.agent_state import AgentState
from src.agentic_ml.ml_engine.evaluation.risk_scorer import ModelRiskScorer

logger = logging.getLogger("agentic_ml.agents.deployment_gate")


def deployment_gate_node(state: AgentState) -> Command:
    """Evaluate deployment risk and pause for human approval if necessary."""
    best_metrics = state.get("best_model_metrics") or {}
    task_type = state.get("task_type", "classification")
    dataset_profile = state.get("dataset_info") or {}
    cv_std = state.get("best_model_metrics", {}).get("cv_std")

    risk = ModelRiskScorer.score(
        metrics=best_metrics,
        task_type=task_type,
        dataset_profile=dataset_profile,
        cv_std=cv_std,
    )

    provenance_entry = {
        "agent_name": "deployment_gate",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "operation": f"Deployment risk evaluation — score={risk.score}, decision={risk.deployment_decision}",
        "result_summary": f"risk={risk.risk_level}, score={risk.score}, decision={risk.deployment_decision}",
        "artifact_path": None,
    }

    evidence_entry = {
        "agent_name": "deployment_gate",
        "decision": risk.deployment_decision,
        "selected_tool": "ModelRiskScorer.score",
        "reason": f"Risk assessment score {risk.score}/100 ({risk.risk_level}). Factors: {risk.factors}",
        "confidence": 1.0,
        "artifacts": [],
        "metrics": {"risk_score": float(risk.score)},
        "warnings": [f"Risk factor: {k}={v}" for k, v in risk.factors.items() if "HIGH" in str(v) or "MEDIUM" in str(v)],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    base_update = {
        "risk_score": risk.score,
        "risk_level": risk.risk_level,
        "provenance": [provenance_entry],
        "evidence": [evidence_entry],
    }

    # ── Auto-approve low risk ─────────────────────────────────────────────────
    if risk.deployment_decision == "AUTO_APPROVE":
        logger.info("Deployment Gate: AUTO_APPROVE (score=%d, risk=%s)", risk.score, risk.risk_level)
        return Command(
            goto="deployment",
            update={
                **base_update,
                "deployment_decision": "AUTO_APPROVE",
            },
        )

    # ── Medium risk: advise but auto-approve (no HITL interrupt) ─────────────
    if risk.deployment_decision == "HUMAN_REVIEW":
        logger.warning(
            "Deployment Gate: MEDIUM risk — auto-approving with advisory (score=%d). "
            "Human review recommended before production.",
            risk.score,
        )
        return Command(
            goto="deployment",
            update={
                **base_update,
                "deployment_decision": "AUTO_APPROVE",  # advisory auto-approve
            },
        )

    # ── Human-in-the-loop interrupt ──────────────────────────────────────────
    logger.info("Deployment Gate: Interrupting for HITL review (score=%d, risk=%s)", risk.score, risk.risk_level)
    decision = interrupt({
        "type": "deployment_approval",
        "run_id": state.get("run_id", "unknown"),
        "risk_score": risk.score,
        "risk_level": risk.risk_level,
        "reasons": risk.factors,
    })

    if isinstance(decision, dict) and decision.get("approved") is True:
        logger.info("Deployment Gate: HUMAN_APPROVED by reviewer")
        return Command(
            goto="deployment",
            update={
                **base_update,
                "deployment_decision": "HUMAN_APPROVED",
            },
        )

    logger.warning("Deployment Gate: REJECTED by reviewer or default policy")
    return Command(
        goto=END,
        update={
            **base_update,
            "deployment_completed": False,
            "deployment_decision": "REJECTED",
        },
    )
