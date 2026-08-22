"""
EDA Agent Node.

Executes automated exploratory data analysis: univariate distributions,
skewness, kurtosis, outlier boundaries, and multi-collinearity detection.
Sets eda_completed=True only after EDAEngine.analyze() returns non-empty stats.
"""
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from src.agentic_ml.state.agent_state import AgentState
from src.agentic_ml.ml_engine.data.loader import DataLoader
from src.agentic_ml.ml_engine.eda.statistics import EDAEngine

logger = logging.getLogger("agentic_ml.agents.eda")

SYSTEM_PROMPT = (
    "You are the Senior EDA & Statistical Profiling Agent. "
    "Analyze univariate distributions, skewness, kurtosis, outlier boundaries "
    "(1.5×IQR), and multi-collinearity (|r| ≥ 0.75). "
    "Provide an actionable technical synthesis for downstream Feature Engineering "
    "and Selection agents."
)


def eda_node(state: AgentState) -> Dict[str, Any]:
    from src.agentic_ml.llm.factory import get_llm
    llm = get_llm()

    task_type = state.get("task_type", "classification")
    dataset_path = state.get("dataset_path", "")

    # Deterministic operations — must succeed before flag is set.
    df, target_col = DataLoader.load_or_synthesize(task_type, dataset_path)
    stats_data = EDAEngine.analyze(df)

    if not stats_data:
        raise RuntimeError("EDAEngine.analyze() returned empty result — eda_completed NOT set.")

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    skewness_map = {
        col: round(float(df[col].skew()), 4)
        for col in numeric_cols
        if len(df[col].dropna()) > 2
    }

    summary_report: Dict[str, Any] = {
        "dimensions": f"{df.shape[0]} rows × {df.shape[1]} columns",
        "numeric_features": len(numeric_cols),
        "skewness": skewness_map,
        "summary_statistics": stats_data.get("summary_stats", {}),
    }

    logger.info("EDA: analyzed %d cols, %d numeric.", df.shape[1], len(numeric_cols))

    execution_mode = state.get("execution_mode", "simulation")
    try:
        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Task: {state.get('current_task', 'ML Modeling')}\n"
                    f"Target Column: {target_col}\n"
                    f"Computed Statistical Data:\n"
                    f"```json\n{json.dumps(summary_report, indent=2)}\n```\n\n"
                    "Synthesize the exploratory findings and highlight feature distribution anomalies."
                )
            ),
        ])
        execution_mode = "live"
    except Exception as exc:
        response = AIMessage(
            content=(
                f"[EDA Agent — Simulation Mode]\n"
                f"- Analyzed {df.shape[0]} rows × {df.shape[1]} attributes.\n"
                f"- Skewed (|skew| > 1.0): {[k for k, v in skewness_map.items() if abs(v) > 1.0]}\n"
                f"- LLM unavailable: {exc}"
            )
        )
        logger.warning("EDA: simulation fallback — %s", exc)

    provenance_entry = {
        "agent_name": "eda",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "operation": "Univariate stats, skewness, kurtosis, multi-collinearity detection",
        "result_summary": f"dims={summary_report['dimensions']}, numeric={len(numeric_cols)}",
        "artifact_path": None,
    }

    return {
        "messages": [response],
        "data_summary": summary_report,
        "eda_completed": True,
        "execution_mode": execution_mode,
        "provenance": [provenance_entry],
    }
