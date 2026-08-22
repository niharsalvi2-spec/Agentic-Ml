"""
Data Collector Agent Node.

Loads or synthesizes the training dataset, profiles schema and statistics.
Sets data_collected=True ONLY after deterministic DataLoader + DataProfiler
operations complete successfully and produce a non-empty profile dict.
"""
import logging
from datetime import datetime, timezone

from langchain_core.messages import SystemMessage, HumanMessage

from src.agentic_ml.state.agent_state import AgentState
from src.agentic_ml.ml_engine.data.loader import DataLoader
from src.agentic_ml.ml_engine.data.profiler import DataProfiler

logger = logging.getLogger("agentic_ml.agents.data_collector")

SYSTEM_PROMPT = (
    "You are the Data Collector Agent. "
    "Summarize dataset acquisition, schema properties, "
    "class distribution, and missingness at a technical level."
)


def data_collector_node(state: AgentState) -> dict:
    from src.agentic_ml.llm.factory import get_llm
    llm = get_llm()

    task_type = state.get("task_type", "classification")
    path = state.get("dataset_path", "")

    # Deterministic operation — must succeed before flag is set.
    df, target_col = DataLoader.load_or_synthesize(task_type, path)
    profile = DataProfiler.profile(df, target_col)

    # Verify profile is non-trivial before declaring completion.
    if not profile or profile.get("n_rows", 0) == 0:
        raise RuntimeError("DataProfiler returned empty profile — data_collected NOT set.")

    logger.info("Data Collector: loaded %d rows, %d cols.", profile["n_rows"], profile["n_columns"])

    execution_mode = state.get("execution_mode", "simulation")
    try:
        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Dataset profiled: {profile['n_rows']} rows, "
                    f"{profile['n_columns']} cols. Target: {target_col}"
                )
            ),
        ])
        execution_mode = "live"
    except Exception as exc:
        from langchain_core.messages import AIMessage
        response = AIMessage(
            content=(
                f"[Data Collector — Simulation Mode]\n"
                f"Loaded {profile['n_rows']} rows × {profile['n_columns']} cols.\n"
                f"Target column: {target_col}\n"
                f"LLM unavailable: {exc}"
            )
        )
        logger.warning("Data Collector: simulation fallback — %s", exc)

    provenance_entry = {
        "agent_name": "data_collector",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "operation": "Loaded dataset and profiled schema",
        "result_summary": f"rows={profile['n_rows']}, cols={profile['n_columns']}, target={target_col}",
        "artifact_path": None,
    }

    return {
        "messages": [response],
        "target_column": target_col,
        "dataset_info": profile,
        "data_collected": True,
        "execution_mode": execution_mode,
        "provenance": [provenance_entry],
    }
