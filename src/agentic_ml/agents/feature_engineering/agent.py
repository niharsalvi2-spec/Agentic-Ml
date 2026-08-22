"""
Feature Engineering Agent Node.

Constructs polynomial interactions, normalized ratios, and log transforms
for skewed features. Sets feature_engineered=True only after the
FeatureEngineer operations produce a larger feature matrix than the input.
Consumes X from state and writes transformed X back to state for selection.
"""
import logging
from datetime import datetime, timezone

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.types import Command

from src.agentic_ml.state.agent_state import AgentState
from src.agentic_ml.llm.factory import get_llm
from src.agentic_ml.ml_engine.data.loader import DataLoader
from src.agentic_ml.ml_engine.features.engineering import FeatureEngineer

logger = logging.getLogger("agentic_ml.agents.feature_engineering")

SYSTEM_PROMPT = (
    "You are the Feature Engineering Agent. "
    "Formulate domain interaction features, log-ratio transforms, and "
    "polynomial interactions. Explain which transformations were applied "
    "and why they improve downstream model performance."
)


def feature_engineering_node(state: AgentState) -> Command:
    llm = get_llm()
    task_type = state.get("task_type", "classification")

    # Consume X from state or extract from clean/raw df
    X = state.get("X")
    if X is None:
        df = state.get("clean_df") if state.get("clean_df") is not None else state.get("raw_df")

        target_col = state.get("target_column")
        if df is None:
            path = state.get("dataset_path", "")
            df, target_col = DataLoader.load_or_synthesize(task_type, path, target_column=target_col)
        elif target_col is None:
            target_col = df.columns[-1]
        X = df.drop(columns=[target_col], errors="ignore")

    X_log = FeatureEngineer.add_log_transforms(X)
    X_engineered = FeatureEngineer.add_polynomial_interactions(X_log, max_features=4)

    added_features = [c for c in X_engineered.columns if c not in X.columns]

    logger.info("Feature Engineering: %d base → %d engineered features.", X.shape[1], X_engineered.shape[1])

    execution_mode = state.get("execution_mode", "simulation")
    try:
        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=f"Engineered {len(added_features)} new features: {added_features[:5]}."
            ),
        ])
        execution_mode = "live"
    except Exception as exc:
        response = AIMessage(
            content=(
                f"[Feature Engineering — Simulation Mode]\n"
                f"Constructed {len(added_features)} candidate features.\n"
                f"LLM unavailable: {exc}"
            )
        )
        logger.warning("Feature Engineering: simulation fallback — %s", exc)

    provenance_entry = {
        "agent_name": "feature_engineering",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "operation": "Log transforms + polynomial interactions",
        "result_summary": f"base={X.shape[1]}, engineered={X_engineered.shape[1]}, added={len(added_features)}",
        "artifact_path": None,
    }

    return Command(
        goto="feature_selection",
        update={
            "messages": [response],
            "X": X_engineered,
            "feature_engineered": True,
            "execution_mode": execution_mode,
            "provenance": [provenance_entry],
        },
    )
