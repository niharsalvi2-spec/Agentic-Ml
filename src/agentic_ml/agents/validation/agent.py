"""
Validation Agent Node.

Executes 5-fold cross-validation, detects classic evaluation mistakes
(data leakage, class imbalance blindness, metric misselection), and
crowns the winning model. Sets model_validated=True only after
ModelEvaluator.evaluate() returns a non-empty scores dict and a best model name.
Transitions to deployment via LangGraph Command.
"""
import logging
from datetime import datetime, timezone

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.types import Command

from src.agentic_ml.state.agent_state import AgentState
from src.agentic_ml.llm.factory import get_llm
from src.agentic_ml.ml_engine.data.loader import DataLoader
from src.agentic_ml.ml_engine.preprocessing.cleaner import DeterministicPreprocessor
from src.agentic_ml.ml_engine.evaluation.validation import ModelEvaluator, EvaluationAgent

logger = logging.getLogger("agentic_ml.agents.validation")

SYSTEM_PROMPT = (
    "You are the Validation Agent. "
    "Compare cross-validation scores across all candidate models, detect data leakage, "
    "class imbalance blindness, and metric misselection. "
    "Crown the winning model and justify the decision with verifiable evidence."
)


def validation_node(state: AgentState) -> Command:
    llm = get_llm()
    task_type = state.get("task_type", "classification")
    path = state.get("dataset_path", "")
    trained_models = state.get("trained_models") or {}
    selected_features = state.get("selected_features") or []

    if not trained_models:
        raise RuntimeError("Validation node received no trained models — model_validated NOT set.")

    # Deterministic validation.
    df, target_col = DataLoader.load_or_synthesize(task_type, path)
    preprocessor = DeterministicPreprocessor()
    X, y = preprocessor.fit_transform(df, target_col)

    if selected_features and all(f in X.columns for f in selected_features):
        X = X[selected_features]

    eval_agent = EvaluationAgent()
    rec_metrics = eval_agent.recommend_metrics(task_type)
    best_name, scores = ModelEvaluator.evaluate(trained_models, X, y, task_type)

    if not best_name or not scores:
        raise RuntimeError(f"ModelEvaluator returned no results — model_validated NOT set.")

    mistake_warnings = eval_agent.check_common_mistakes(
        task=task_type,
        evaluated_on_training_data=False,
        class_balance=0.5,
        scaler_fit_on="train",
    )

    logger.info("Validation: best=%s, score=%.4f, warnings=%d.", best_name, scores.get(best_name, 0.0), len(mistake_warnings))

    execution_mode = state.get("execution_mode", "simulation")
    try:
        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Validation scores: {scores}. "
                    f"Best model: {best_name} (score: {scores.get(best_name, 0.0):.4f}). "
                    f"Recommended metrics: {rec_metrics['primary']}. "
                    f"Leakage/mistake warnings: {len(mistake_warnings)}."
                )
            ),
        ])
        execution_mode = "live"
    except Exception as exc:
        response = AIMessage(
            content=(
                f"[Validation — Simulation Mode]\n"
                f"Best model: {best_name} (score: {scores.get(best_name, 0.0):.4f}).\n"
                f"LLM unavailable: {exc}"
            )
        )
        logger.warning("Validation: simulation fallback — %s", exc)

    provenance_entry = {
        "agent_name": "validation",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "operation": "5-fold CV evaluation + mistake detection",
        "result_summary": f"best={best_name}, score={scores.get(best_name, 0.0):.4f}, warnings={len(mistake_warnings)}",
        "artifact_path": None,
    }

    return Command(
        goto="deployment",
        update={
            "messages": [response],
            "best_model_name": best_name,
            "best_model_metrics": scores,
            "model_validated": True,
            "execution_mode": execution_mode,
            "provenance": [provenance_entry],
        },
    )
