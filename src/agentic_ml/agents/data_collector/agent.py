"""
Data Collector Agent Node.

Loads or synthesizes the training dataset, profiles schema and statistics.
Sets data_collected=True ONLY after deterministic DataLoader + DataProfiler
operations complete successfully and produce a non-empty profile dict.
Transitions to preprocessing via LangGraph Command, storing raw_df in state.
"""
import logging
from datetime import datetime, timezone

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.types import Command

from src.agentic_ml.state.agent_state import AgentState
from src.agentic_ml.llm.factory import get_llm
from src.agentic_ml.ml_engine.data.loader import DataLoader
from src.agentic_ml.ml_engine.data.profiler import DataProfiler

logger = logging.getLogger("agentic_ml.agents.data_collector")

SYSTEM_PROMPT = (
    "You are the Data Collector Agent. "
    "Summarize dataset acquisition, schema properties, "
    "class distribution, and missingness at a technical level."
)


def data_collector_node(state: AgentState) -> Command:
    llm = get_llm()
    task_type = state.get("task_type", "classification")
    path = state.get("dataset_path", "")
    target_col_req = state.get("target_column")

    # Deterministic operation — load data with seeded reproducibility
    random_seed = state.get("random_seed", 42)
    df, target_col = DataLoader.load_or_synthesize(
        task_type=task_type,
        dataset_path=path,
        target_column=target_col_req,
        random_state=random_seed,
    )
    profile = DataProfiler.profile(df, target_col)

    # Compute class distribution & minority percentage for classification tasks
    if task_type == "classification" and target_col in df.columns:
        counts = df[target_col].value_counts(normalize=True)
        profile["class_distribution"] = counts.to_dict()
        profile["minority_pct"] = float(counts.min() * 100.0) if len(counts) > 0 else 50.0
    profile["row_count"] = profile.get("n_rows", len(df))
    profile["column_count"] = profile.get("n_columns", len(df.columns))

    # Verify profile is non-trivial before declaring completion
    if not profile or profile.get("n_rows", 0) == 0:
        raise RuntimeError("DataProfiler returned empty profile — data_collected NOT set.")

    logger.info("Data Collector: loaded %d rows, %d cols, target='%s', seed=%d.", profile["n_rows"], profile["n_columns"], target_col, random_seed)

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
        "operation": "Loaded dataset, verified schema, and profiled missingness",
        "result_summary": f"rows={profile['n_rows']}, cols={profile['n_columns']}, target={target_col}",
        "artifact_path": None,
    }

    return Command(
        goto="preprocessing",
        update={
            "messages": [response],
            "target_column": target_col,
            "dataset_info": profile,
            "raw_df": df,
            "data_collected": True,
            "execution_mode": execution_mode,
            "provenance": [provenance_entry],
        },
    )
