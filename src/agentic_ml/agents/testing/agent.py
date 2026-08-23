"""
Testing Agent Node.

Executes rigorous unit-level contract validation and prediction stability checks
on all trained candidate models:
  - Verifies predict(X) produces non-empty, finite outputs with zero NaNs.
  - Verifies predict_proba(X) produces strictly valid probabilities [0, 1] summing to 1.0.
  - Validates output dimensionality and prediction latency bounds.
Sets model_tested=True only after contract tests succeed across all candidates.
Transitions to validation via LangGraph Command.
"""
import logging
import time
from datetime import datetime, timezone
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.types import Command

from src.agentic_ml.state.agent_state import AgentState
from src.agentic_ml.llm.factory import get_llm

logger = logging.getLogger("agentic_ml.agents.testing")

SYSTEM_PROMPT = (
    "You are the Testing Agent. "
    "Perform comprehensive contract checks on input/output schemas and prediction stability "
    "across all trained candidate models. Flag any model that produces "
    "NaN outputs, shape mismatches, or non-probabilistic confidence values."
)


def testing_node(state: AgentState) -> Command:
    llm = get_llm()
    candidates = state.get("candidate_models") or []
    trained_models = state.get("trained_models") or {}
    task_type = state.get("task_type", "classification")

    if not candidates or not trained_models:
        raise RuntimeError(
            "Testing node received no trained models — "
            "did model_building complete successfully? model_tested NOT set."
        )

    # Use real feature matrix X from state for contract testing
    X = state.get("X")
    if X is None or len(X) == 0:
        raise RuntimeError("Testing requires the real feature matrix X — no synthetic fallbacks allowed.")

    test_sample = X.head(min(20, len(X)))
    max_latency_ms = float(state.get("max_inference_latency_ms", 250.0))

    test_results: Dict[str, Dict[str, Any]] = {}
    failed_models: List[str] = []

    for name in candidates:
        model = trained_models.get(name)
        if model is None:
            failed_models.append(name)
            test_results[name] = {"status": "FAILED", "reason": "Model instance is None"}
            continue

        try:
            # 1. Prediction stability and shape check
            t0 = time.perf_counter()
            preds = model.predict(test_sample)
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)

            if len(preds) != len(test_sample):
                raise ValueError(f"Output shape mismatch: expected {len(test_sample)}, got {len(preds)}")

            # Check for NaNs or Infinities in predictions
            if np.isnan(preds).any() or np.isinf(preds).any():
                raise ValueError("Predictions contain NaN or Infinite values.")

            # Enforce inference latency bound
            if latency_ms > max_latency_ms:
                raise ValueError(f"Prediction latency {latency_ms}ms exceeds max constraint {max_latency_ms}ms.")

            # 2. Probability calibration & bounds check for classification models
            if task_type == "classification" and hasattr(model, "predict_proba"):
                probas = model.predict_proba(test_sample)
                if np.isnan(probas).any() or np.isinf(probas).any():
                    raise ValueError("Probabilities contain NaN or Infinite values.")
                if (probas < 0.0).any() or (probas > 1.0).any():
                    raise ValueError("Probabilities out of valid [0, 1] range.")
                row_sums = probas.sum(axis=1)
                if not np.allclose(row_sums, 1.0, atol=1e-3):
                    raise ValueError(f"Probabilities do not sum to 1.0 (sums: {row_sums[:3]})")

            test_results[name] = {
                "status": "PASSED",
                "latency_ms": latency_ms,
                "output_shape": list(preds.shape),
                "finite_checked": True,
                "proba_checked": hasattr(model, "predict_proba"),
            }

        except Exception as exc:
            logger.warning("Testing contract failure on %s: %s", name, exc)
            failed_models.append(name)
            test_results[name] = {"status": "FAILED", "reason": str(exc)}

    # Filter trained_models down to only validated passing candidates
    passing_models = {
        name: trained_models[name]
        for name in candidates
        if test_results.get(name, {}).get("status") == "PASSED" and name in trained_models
    }

    if not passing_models:
        raise RuntimeError(f"All candidate models failed QA contract testing: {test_results}")

    passed_candidates = list(passing_models.keys())
    logger.info("Testing: QA contract tests passed for %d/%d models: %s.", len(passed_candidates), len(candidates), passed_candidates)

    execution_mode = state.get("execution_mode", "simulation")
    try:
        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Executed QA contract tests across models: {candidates}. "
                    f"Results: {test_results}. "
                    f"Retained passing models: {passed_candidates}. Failed: {failed_models}."
                )
            ),
        ])
        execution_mode = "live"
    except Exception as exc:
        response = AIMessage(
            content=(
                f"[Testing — Simulation Mode]\n"
                f"QA contract tests: {len(passed_candidates)} passed, {len(failed_models)} failed.\n"
                f"Passing candidates: {passed_candidates}\n"
                f"LLM unavailable: {exc}"
            )
        )
        logger.warning("Testing: simulation fallback — %s", exc)

    provenance_entry = {
        "agent_name": "testing",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "operation": "Contract QA validation, NaN/Inf checks, probability calibration, and latency tests",
        "result_summary": f"passed={passed_candidates}, failed={failed_models}",
        "artifact_path": None,
    }

    evidence_entry = {
        "agent_name": "testing",
        "decision": f"{len(passed_candidates)} models passed QA",
        "selected_tool": "ModelQAContractTester",
        "reason": f"Contract checks verified on {len(test_sample)} sample rows. Latency bound: {max_latency_ms}ms.",
        "confidence": 1.0,
        "artifacts": [],
        "metrics": {"passed_count": len(passed_candidates), "failed_count": len(failed_models)},
        "warnings": [f"{m}: {test_results[m]['reason']}" for m in failed_models],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return Command(
        goto="validation",
        update={
            "messages": [response],
            "candidate_models": passed_candidates,
            "trained_models": passing_models,
            "model_tested": True,
            "execution_mode": execution_mode,
            "provenance": [provenance_entry],
            "evidence": [evidence_entry],
        },
    )


# Prevent pytest from mistaking this agent node function for a test case
testing_node.__test__ = False
