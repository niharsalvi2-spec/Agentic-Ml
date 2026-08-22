"""
Preprocessing Agent Node.

Executes leakage-safe missing value imputation, IQR outlier fence clipping,
and StandardScaler normalization. Sets data_preprocessed=True only after the
DeterministicPreprocessor fit_transform() call completes and returns a
non-empty feature matrix.
"""
import logging
from datetime import datetime, timezone

from langchain_core.messages import SystemMessage, HumanMessage

from src.agentic_ml.state.agent_state import AgentState
from src.agentic_ml.ml_engine.data.loader import DataLoader
from src.agentic_ml.ml_engine.preprocessing.cleaner import DeterministicPreprocessor, missingness_report

logger = logging.getLogger("agentic_ml.agents.preprocessing")

SYSTEM_PROMPT = (
    "You are the Preprocessing Agent. "
    "Report on missingness patterns (MCAR/MAR evidence), outlier fence clipping, "
    "and feature scaling strategy applied to the dataset."
)


def preprocessing_node(state: AgentState) -> dict:
    from src.agentic_ml.llm.factory import get_llm
    llm = get_llm()

    task_type = state.get("task_type", "classification")
    path = state.get("dataset_path", "")

    # Deterministic operations — both must succeed before flag is set.
    df, target_col = DataLoader.load_or_synthesize(task_type, path)
    miss_rep = missingness_report(df).to_dict(orient="index")

    preprocessor = DeterministicPreprocessor(clip_outliers=True)
    X, y = preprocessor.fit_transform(df, target_col)

    if X is None or X.shape[0] == 0:
        raise RuntimeError("DeterministicPreprocessor returned empty matrix — data_preprocessed NOT set.")

    logger.info("Preprocessing: %s → %s after cleaning.", df.shape, X.shape)

    execution_mode = state.get("execution_mode", "simulation")
    try:
        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Preprocessed dataset with target '{target_col}'. "
                    f"Input shape: {df.shape} → Processed: {X.shape}. "
                    f"Missingness columns handled: {len(miss_rep)}."
                )
            ),
        ])
        execution_mode = "live"
    except Exception as exc:
        from langchain_core.messages import AIMessage
        response = AIMessage(
            content=(
                f"[Preprocessing — Simulation Mode]\n"
                f"Input {df.shape} → Output {X.shape}.\n"
                f"Missingness addressed in {len(miss_rep)} columns.\n"
                f"LLM unavailable: {exc}"
            )
        )
        logger.warning("Preprocessing: simulation fallback — %s", exc)

    provenance_entry = {
        "agent_name": "preprocessing",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "operation": "Leakage-safe imputation, outlier clipping, StandardScaler normalisation",
        "result_summary": f"input={df.shape}, output={X.shape}, missing_cols={len(miss_rep)}",
        "artifact_path": None,
    }

    return {
        "messages": [response],
        "target_column": target_col,
        "data_preprocessed": True,
        "execution_mode": execution_mode,
        "provenance": [provenance_entry],
    }
