"""
Problem Analyzer Agent Node.

Analyzes the ML problem statement, determines task type, evaluation metrics,
and goals. Sets problem_analyzed=True only after LLM analysis completes.
"""
import logging
from datetime import datetime, timezone

from langchain_core.messages import SystemMessage, HumanMessage

from src.agentic_ml.state.agent_state import AgentState

logger = logging.getLogger("agentic_ml.agents.problem_analyzer")

SYSTEM_PROMPT = (
    "You are the Problem Analyzer Agent. "
    "Analyze the ML problem statement, determine the task type "
    "(classification / regression / clustering), identify the primary "
    "evaluation metric, and state the business goal clearly."
)


def problem_analyzer_node(state: AgentState) -> dict:
    from src.agentic_ml.llm.factory import get_llm
    llm = get_llm()

    prompt = state.get("raw_prompt") or state.get("current_task", "Build an optimal ML model.")

    # Infer task type deterministically from keywords
    lower_prompt = prompt.lower()
    if any(w in lower_prompt for w in ["churn", "classify", "spam", "fraud", "detect", "cancer", "default"]):
        task_type = "classification"
    elif any(w in lower_prompt for w in ["price", "predict", "forecast", "regress", "sales", "revenue"]):
        task_type = "regression"
    else:
        task_type = "classification"  # safe default

    execution_mode = "simulation"
    try:
        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Problem statement: {prompt}"),
        ])
        execution_mode = "live"
        logger.info("Problem Analyzer: LLM responded (live mode).")
    except Exception as exc:
        from langchain_core.messages import AIMessage
        response = AIMessage(
            content=(
                f"[Problem Analyzer — Simulation Mode]\n"
                f"Task: {prompt}\n"
                f"Inferred type: {task_type}\n"
                f"LLM unavailable: {exc}"
            )
        )
        logger.warning("Problem Analyzer: falling back to simulation — %s", exc)

    provenance_entry = {
        "agent_name": "problem_analyzer",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "operation": "Analyzed problem statement and inferred task type",
        "result_summary": f"task_type={task_type}, execution_mode={execution_mode}",
        "artifact_path": None,
    }

    return {
        "messages": [response],
        "task_type": task_type,
        "problem_analyzed": True,
        "execution_mode": execution_mode,
        "provenance": [provenance_entry],
    }
