"""
Feature Selection Agent Node.

Performs statistical ANOVA, Mutual Information, and Tree Importance feature
selection to retain only high-signal features. Sets feature_selection_completed=True
only after FeatureSelector.select_top_k() returns a non-empty feature list.
Consumes X and y from state and updates X with selected features for model training.
"""
import logging
from datetime import datetime, timezone

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.types import Command

from src.agentic_ml.state.agent_state import AgentState
from src.agentic_ml.llm.factory import get_llm
from src.agentic_ml.ml_engine.data.loader import DataLoader
from src.agentic_ml.ml_engine.preprocessing.cleaner import DeterministicPreprocessor
from src.agentic_ml.ml_engine.features.selection import FeatureSelector

logger = logging.getLogger("agentic_ml.agents.feature_selection")

SYSTEM_PROMPT = (
    "You are the Feature Selection Agent. "
    "Filter out redundant noise features based on ANOVA F-statistics, "
    "mutual information scores, and tree-based importance ranking. "
    "Justify which features were retained and which were dropped."
)


def feature_selection_node(state: AgentState) -> Command:
    llm = get_llm()
    task_type = state.get("task_type", "classification")

    # Consume X and y from state
    X = state.get("X")
    y = state.get("y")

    if X is None or y is None:
        df = state.get("clean_df") if state.get("clean_df") is not None else state.get("raw_df")

        target_col = state.get("target_column")
        if df is None:
            path = state.get("dataset_path", "")
            df, target_col = DataLoader.load_or_synthesize(task_type, path, target_column=target_col)
        preprocessor = state.get("preprocessor_obj") or DeterministicPreprocessor()
        X, y = preprocessor.fit_transform(df, target_col)

    k = min(4, X.shape[1])
    selected = FeatureSelector.select_top_k(X, y, task_type=task_type, k=k)

    if not selected:
        raise RuntimeError("FeatureSelector returned empty selection — feature_selection_completed NOT set.")

    # Slice X to selected features
    X_selected = X[selected]

    logger.info("Feature Selection: %d candidate → %d selected features: %s.", X.shape[1], len(selected), selected)

    execution_mode = state.get("execution_mode", "simulation")
    try:
        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Evaluated {X.shape[1]} candidate features. "
                    f"Retained top {len(selected)} high-signal features: {selected}."
                )
            ),
        ])
        execution_mode = "live"
    except Exception as exc:
        response = AIMessage(
            content=(
                f"[Feature Selection — Simulation Mode]\n"
                f"Retained top {len(selected)} features: {selected}.\n"
                f"LLM unavailable: {exc}"
            )
        )
        logger.warning("Feature Selection: simulation fallback — %s", exc)

    provenance_entry = {
        "agent_name": "feature_selection",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "operation": "ANOVA / Mutual Info / Tree Importance top-k selection",
        "result_summary": f"candidates={X.shape[1]}, selected={len(selected)}, features={selected}",
        "artifact_path": None,
    }

    return Command(
        goto="model_building",
        update={
            "messages": [response],
            "X": X_selected,
            "y": y,
            "selected_features": selected,
            "feature_selection_completed": True,
            "execution_mode": execution_mode,
            "provenance": [provenance_entry],
        },
    )
