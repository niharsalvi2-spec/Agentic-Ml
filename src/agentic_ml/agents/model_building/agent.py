"""
Model Building Agent Node.

Trains diverse candidate model families (Logistic Regression, Random Forest,
XGBoost, etc.) and benchmarks performance. Sets model_built=True only after
ModelTrainer.train_candidates() returns a non-empty dict of fitted models.
Transitions to testing via LangGraph Command.
"""
import logging
from datetime import datetime, timezone

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.types import Command

from src.agentic_ml.state.agent_state import AgentState
from src.agentic_ml.llm.factory import get_llm
from src.agentic_ml.ml_engine.data.loader import DataLoader
from src.agentic_ml.ml_engine.preprocessing.cleaner import DeterministicPreprocessor
from src.agentic_ml.ml_engine.models.registry import ModelRegistry
from src.agentic_ml.ml_engine.models.training import ModelTrainer

logger = logging.getLogger("agentic_ml.agents.model_building")

SYSTEM_PROMPT = (
    "You are the Model Building Agent. "
    "Train diverse algorithm families and evaluate architectures. "
    "Explain the rationale for choosing each candidate family given the task type, "
    "dataset size, and feature dimensionality."
)


def model_building_node(state: AgentState) -> Command:
    llm = get_llm()
    task_type = state.get("task_type", "classification")
    path = state.get("dataset_path", "")
    selected_features = state.get("selected_features") or []

    # Deterministic operations.
    df, target_col = DataLoader.load_or_synthesize(task_type, path)
    preprocessor = DeterministicPreprocessor()
    X, y = preprocessor.fit_transform(df, target_col)

    # Restrict to selected features if they are all present in the matrix.
    if selected_features and all(f in X.columns for f in selected_features):
        X = X[selected_features]

    recommendations = ModelRegistry.recommend(
        task_type=task_type,
        n_samples=len(X),
        n_features=X.shape[1],
        need_interpretability=True,
        need_proba=True,
        suspect_nonlinear=True,
    )

    trained_models = ModelTrainer.train_candidates(X, y, task_type)
    candidates = list(trained_models.keys())

    if not trained_models:
        raise RuntimeError("ModelTrainer returned no fitted models — model_built NOT set.")

    logger.info("Model Building: trained %d candidates: %s.", len(candidates), candidates)

    execution_mode = state.get("execution_mode", "simulation")
    try:
        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Trained {len(candidates)} model candidates: {candidates}. "
                    f"Recommendations: {recommendations[:2]}."
                )
            ),
        ])
        execution_mode = "live"
    except Exception as exc:
        response = AIMessage(
            content=(
                f"[Model Building — Simulation Mode]\n"
                f"Trained {len(candidates)} baselines: {candidates}.\n"
                f"LLM unavailable: {exc}"
            )
        )
        logger.warning("Model Building: simulation fallback — %s", exc)

    provenance_entry = {
        "agent_name": "model_building",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "operation": "Multi-family model training",
        "result_summary": f"candidates={candidates}",
        "artifact_path": None,
    }

    return Command(
        goto="testing",
        update={
            "messages": [response],
            "candidate_models": candidates,
            "trained_models": trained_models,
            "model_built": True,
            "execution_mode": execution_mode,
            "provenance": [provenance_entry],
        },
    )
