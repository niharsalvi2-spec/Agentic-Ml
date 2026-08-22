"""
Problem Analyzer Agent Node.

Analyzes the ML problem statement, determines task type, evaluation metrics,
and goals. Sets problem_analyzed=True only after analysis completes.

Key design decisions:
  1. task_type is inferred from keywords BUT recorded with confidence + method in evidence.
     It is NEVER silently assumed — the inference method is always provenance-tracked.
  2. target_column: only set if user explicitly provided it in the request.
     If None, target_inference_method = "last_column" is recorded and the column
     will be inferred by data_collector after seeing the actual dataset.
  3. The LLM is used for richer reasoning — but task_type is set deterministically
     from keywords as a fallback. LLM does NOT set task_type without validation.

Transitions to data_collector via LangGraph Command.
"""
import logging
import re
from datetime import datetime, timezone
from typing import Optional

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.types import Command

from src.agentic_ml.state.agent_state import AgentState
from src.agentic_ml.llm.factory import get_llm

logger = logging.getLogger("agentic_ml.agents.problem_analyzer")

SYSTEM_PROMPT = (
    "You are the Problem Analyzer Agent for an agentic ML platform. "
    "Analyze the problem statement and output a structured analysis covering:\n"
    "1. Task type: classification, regression, or clustering\n"
    "2. Primary evaluation metric(s)\n"
    "3. Business goal and success criteria\n"
    "4. Key constraints (latency, interpretability, fairness)\n"
    "5. Any assumptions about the data\n\n"
    "Be precise and evidence-based. Do not fabricate metrics — state only what the "
    "problem statement specifies."
)

# Keyword sets for deterministic task type inference
_CLASSIFICATION_KEYWORDS = {
    "churn", "classify", "classification", "spam", "fraud", "detect", "detection",
    "cancer", "default", "approve", "approval", "predict probability", "binary",
    "multi-class", "sentiment", "diagnostic", "risk score",
}
_REGRESSION_KEYWORDS = {
    "price", "pricing", "predict sales", "forecast", "regression", "revenue",
    "sales", "demand", "estimate", "quantity", "cost", "temperature", "stock",
    "time series", "continuous",
}
_CLUSTERING_KEYWORDS = {
    "cluster", "clustering", "segment", "segmentation", "group", "grouping",
    "unsupervised", "anomaly", "outlier detection",
}


def _infer_task_type(prompt: str) -> tuple[str, float, str]:
    """
    Deterministically infer task type from keywords.

    Returns:
        (task_type, confidence, inference_method)
    """
    lower = prompt.lower()

    classification_hits = sum(1 for kw in _CLASSIFICATION_KEYWORDS if kw in lower)
    regression_hits = sum(1 for kw in _REGRESSION_KEYWORDS if kw in lower)
    clustering_hits = sum(1 for kw in _CLUSTERING_KEYWORDS if kw in lower)

    if clustering_hits > 0:
        return "clustering", min(1.0, 0.6 + clustering_hits * 0.15), "keyword_match"
    if regression_hits > classification_hits:
        return "regression", min(1.0, 0.6 + regression_hits * 0.1), "keyword_match"
    if classification_hits > 0:
        return "classification", min(1.0, 0.6 + classification_hits * 0.1), "keyword_match"

    # No strong signal — default to classification with low confidence
    return "classification", 0.50, "default_fallback"


def problem_analyzer_node(state: AgentState) -> Command:
    llm = get_llm()
    prompt = state.get("raw_prompt") or state.get("current_task", "Build an optimal ML model.")

    # ── Deterministic task type inference ─────────────────────────────────────
    task_type, inference_confidence, inference_method = _infer_task_type(prompt)

    # ── Target column: only use if explicitly provided ────────────────────────
    # Never silently default — record the inference method
    user_target = state.get("target_column")
    if user_target and user_target.strip():
        target_column = user_target.strip()
        target_inference_method = "explicit"
    else:
        target_column = None   # will be set by data_collector after seeing the dataset
        target_inference_method = "last_column" if task_type != "clustering" else "none"

    # ── LLM reasoning (enriches evidence but does NOT override deterministic results) ──
    execution_mode = "simulation"
    try:
        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Problem statement: {prompt}"),
        ])
        execution_mode = "live"
        logger.info("Problem Analyzer: LLM responded (live mode).")
    except Exception as exc:
        response = AIMessage(
            content=(
                f"[Problem Analyzer — Simulation Mode]\n"
                f"Task: {prompt}\n"
                f"Inferred type: {task_type} (confidence={inference_confidence:.0%}, "
                f"method={inference_method})\n"
                f"Target column: {target_column or '(will infer from dataset)'}\n"
                f"LLM unavailable: {exc}"
            )
        )
        logger.warning("Problem Analyzer: falling back to simulation — %s", exc)

    provenance_entry = {
        "agent_name": "problem_analyzer",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "operation": "Problem statement analysis and task type inference",
        "result_summary": (
            f"task_type={task_type}, confidence={inference_confidence:.0%}, "
            f"method={inference_method}, target={target_column or 'defer_to_collector'}, "
            f"target_inference={target_inference_method}"
        ),
        "artifact_path": None,
    }

    evidence_entry = {
        "agent_name": "problem_analyzer",
        "decision": f"task_type={task_type}",
        "selected_tool": "keyword_inference + LLM analysis",
        "reason": (
            f"Keyword matching confidence={inference_confidence:.0%} via {inference_method}. "
            f"Target column: {target_column or 'deferred to data_collector'}."
        ),
        "confidence": inference_confidence,
        "artifacts": [],
        "metrics": {"inference_confidence": inference_confidence},
        "warnings": (
            ["Low confidence task type inference — verify with data_collector"]
            if inference_confidence < 0.65 else []
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return Command(
        goto="data_collector",
        update={
            "messages": [response],
            "task_type": task_type,
            "target_column": target_column,
            "target_inference_method": target_inference_method,
            "problem_analyzed": True,
            "execution_mode": execution_mode,
            "provenance": [provenance_entry],
            "evidence": [evidence_entry],
        },
    )
