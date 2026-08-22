"""
Failure Analyzer Agent Node.

Receives control from the Validation Agent when validation fails.
Diagnoses the root cause using FailureAnalyzer and routes back to
model_building with an adjusted strategy context.

This node makes the graph genuinely agentic — it reasons about failure
and decides whether to retry with a different approach or halt.

Routing:
  - retry_with_different_model  → Command(goto="model_building")
  - retry_with_resampling       → Command(goto="model_building")
  - retry_preprocessing         → Command(goto="preprocessing")
  - flag_and_stop               → Command(goto=END)
"""
import logging
from datetime import datetime, timezone

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.types import Command
from langgraph.graph import END

from src.agentic_ml.state.agent_state import AgentState
from src.agentic_ml.llm.factory import get_llm

logger = logging.getLogger("agentic_ml.agents.failure_analyzer")

SYSTEM_PROMPT = (
    "You are the Failure Analyzer Agent. "
    "You receive validation failure evidence and diagnose the root cause. "
    "Explain why the model failed, what the likely cause is, and what should change "
    "in the retry. Be precise and evidence-based — no generic advice."
)


def failure_analyzer_node(state: AgentState) -> Command:
    llm = get_llm()

    failure_analysis = state.get("last_failure_analysis") or {}
    root_cause = failure_analysis.get("root_cause", "Unknown failure cause")
    action = failure_analysis.get("remediation_action", "retry_with_different_model")
    categories = failure_analysis.get("failure_categories", [])
    retry_count = state.get("validation_retry_count", 1)

    logger.info(
        "FailureAnalyzer: action=%s, categories=%s, retry=%d",
        action, categories, retry_count,
    )

    execution_mode = state.get("execution_mode", "simulation")
    try:
        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Validation failure analysis:\n"
                    f"Root cause: {root_cause}\n"
                    f"Categories: {categories}\n"
                    f"Recommended action: {action}\n"
                    f"Retry #{retry_count}.\n"
                    f"Explain what will change in the next training attempt."
                )
            ),
        ])
        execution_mode = "live"
    except Exception as exc:
        response = AIMessage(
            content=(
                f"[Failure Analyzer — Simulation Mode]\n"
                f"Root cause: {root_cause}\n"
                f"Action: {action}\n"
                f"LLM unavailable: {exc}"
            )
        )
        logger.warning("FailureAnalyzer: simulation fallback — %s", exc)

    provenance_entry = {
        "agent_name": "failure_analyzer",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "operation": f"Failure diagnosis — action={action}",
        "result_summary": f"categories={categories}, action={action}, retry={retry_count}",
        "artifact_path": None,
    }

    evidence_entry = {
        "agent_name": "failure_analyzer",
        "decision": action,
        "selected_tool": "FailureAnalyzer.analyze",
        "reason": root_cause,
        "confidence": 0.8,
        "artifacts": [],
        "metrics": {},
        "warnings": categories,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    base_update = {
        "messages": [response],
        "execution_mode": execution_mode,
        "provenance": [provenance_entry],
        "evidence": [evidence_entry],
    }

    # ── Route based on remediation action ────────────────────────────────────
    if action == "flag_and_stop":
        return Command(
            goto=END,
            update={
                **base_update,
                "errors": [{
                    "agent_name": "failure_analyzer",
                    "error_type": "irrecoverable_failure",
                    "message": root_cause,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "fatal": True,
                }],
            },
        )

    elif action == "retry_preprocessing":
        # Route back to preprocessing with remediation hint
        return Command(
            goto="preprocessing",
            update={
                **base_update,
                # Signal to preprocessing that it should handle imbalance
                "last_failure_analysis": {
                    **failure_analysis,
                    "remediation_hint": "Apply class balancing or review feature scaling.",
                },
            },
        )

    else:
        # retry_with_different_model | retry_with_resampling → model_building
        # Reset model_built so model_building executes fresh
        return Command(
            goto="model_building",
            update={
                **base_update,
                "model_built": False,
                "trained_models": {},
                "best_model_name": None,
                "best_model_metrics": {},
                "last_failure_analysis": {
                    **failure_analysis,
                    "remediation_hint": (
                        "Try different model families. "
                        "If class_imbalance detected, use class_weight='balanced'."
                        if "class_imbalance" in categories or "poor_recall" in categories
                        else "Explore simpler models or regularize aggressively."
                    ),
                },
            },
        )
