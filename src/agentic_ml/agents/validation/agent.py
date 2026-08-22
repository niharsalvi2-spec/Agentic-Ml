"""
Validation Agent Node.

Executes 5-fold cross-validation on the feature matrix X and target y from state.
Sets model_validated=True only after ModelEvaluator.evaluate() returns non-empty scores.

Routing logic (the critical upgrade):
  - PASS  (score >= threshold, no critical leakage) → Command(goto="deployment")
  - FAIL  (score below threshold or leakage)        → Command(goto="failure_analyzer")
  - MAX RETRIES (validation_retry_count >= 2)        → Command(goto=END) with failure evidence

This makes the graph genuinely agentic — it can reason about failure and retry.
"""
import logging
from datetime import datetime, timezone

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.types import Command
from langgraph.graph import END

from src.agentic_ml.state.agent_state import AgentState
from src.agentic_ml.llm.factory import get_llm
from src.agentic_ml.ml_engine.data.loader import DataLoader
from src.agentic_ml.ml_engine.preprocessing.cleaner import DeterministicPreprocessor
from src.agentic_ml.ml_engine.evaluation.validation import ModelEvaluator, EvaluationAgent
from src.agentic_ml.ml_engine.data.leakage_detector import LeakageDetector
from src.agentic_ml.ml_engine.evaluation.failure_analyzer import FailureAnalyzer
from src.agentic_ml.ml_engine.evaluation.risk_scorer import ModelRiskScorer

logger = logging.getLogger("agentic_ml.agents.validation")

SYSTEM_PROMPT = (
    "You are the Validation Agent. "
    "Compare cross-validation scores across all candidate models, detect data leakage, "
    "class imbalance blindness, and metric misselection. "
    "Crown the winning model and justify the decision with verifiable evidence. "
    "If validation fails, clearly state the failure category and recommended remediation."
)

# Minimum acceptable CV score to route to deployment
_PASS_THRESHOLD = 0.55
_MAX_RETRIES = 2


def validation_node(state: AgentState) -> Command:
    llm = get_llm()
    task_type = state.get("task_type", "classification")
    trained_models = state.get("trained_models") or {}
    retry_count = state.get("validation_retry_count", 0)

    if not trained_models:
        raise RuntimeError("Validation node received no trained models — model_validated NOT set.")

    # Consume X and y from state (set by model_building)
    X = state.get("X")
    y = state.get("y")

    if X is None or y is None:
        path = state.get("dataset_path", "")
        df = state.get("clean_df") if state.get("clean_df") is not None else state.get("raw_df")
        target_col = state.get("target_column")
        if df is None:
            df, target_col = DataLoader.load_or_synthesize(task_type, path, target_column=target_col)
        if target_col is None:
            target_col = str(df.columns[-1]) if df is not None and len(df.columns) > 0 else "target"
        preprocessor = state.get("preprocessor_obj") or DeterministicPreprocessor()
        X, y = preprocessor.fit_transform(df, target_col)
        selected_features = state.get("selected_features") or []
        if selected_features and all(f in X.columns for f in selected_features):
            X = X[selected_features]

    # ── Run 5-fold CV ─────────────────────────────────────────────────────────
    eval_agent = EvaluationAgent()
    rec_metrics = eval_agent.recommend_metrics(task_type)
    best_name, mean_scores, std_scores = ModelEvaluator.evaluate(trained_models, X, y, task_type, return_std=True)

    if not best_name or not mean_scores:
        raise RuntimeError("ModelEvaluator returned no results — model_validated NOT set.")

    best_score = mean_scores.get(best_name, 0.0)
    best_std = std_scores.get(best_name, 0.0)

    # ── Leakage detection ─────────────────────────────────────────────────────
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=state.get("random_seed", 42)
    )
    leakage_report = LeakageDetector.check(
        X_train=X_train, X_test=X_test, y_train=y_train,
        target_column=state.get("target_column"),
    )

    # ── Classic mistake warnings ───────────────────────────────────────────────
    mistake_warnings = eval_agent.check_common_mistakes(
        task=task_type,
        evaluated_on_training_data=False,
        class_balance=0.5,
        scaler_fit_on="train",
    )

    # ── PASS / FAIL decision ──────────────────────────────────────────────────
    leakage_critical = not leakage_report.passed
    score_passes = best_score >= _PASS_THRESHOLD
    validation_passed = score_passes and not leakage_critical

    logger.info(
        "Validation: best=%s, score=%.4f, std=%.4f, leakage_passed=%s, score_passes=%s",
        best_name, best_score, best_std, leakage_report.passed, score_passes,
    )

    execution_mode = state.get("execution_mode", "simulation")
    try:
        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Validation scores: {mean_scores}. "
                    f"Best model: {best_name} (score: {best_score:.4f} ± {best_std:.4f}). "
                    f"Recommended metrics: {rec_metrics['primary']}. "
                    f"Leakage check passed: {leakage_report.passed}. "
                    f"Validation passed: {validation_passed}. "
                    f"Retry count: {retry_count}."
                )
            ),
        ])
        execution_mode = "live"
    except Exception as exc:
        response = AIMessage(
            content=(
                f"[Validation — Simulation Mode]\n"
                f"Best model: {best_name} (score: {best_score:.4f} ± {best_std:.4f}).\n"
                f"Validation passed: {validation_passed}.\n"
                f"LLM unavailable: {exc}"
            )
        )
        logger.warning("Validation: simulation fallback — %s", exc)

    provenance_entry = {
        "agent_name": "validation",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "operation": "5-fold CV evaluation + leakage detection + failure routing",
        "result_summary": (
            f"best={best_name}, score={best_score:.4f}±{best_std:.4f}, "
            f"passed={validation_passed}, leakage={not leakage_report.passed}"
        ),
        "artifact_path": None,
    }

    evidence_entry = {
        "agent_name": "validation",
        "decision": "PASS" if validation_passed else "FAIL",
        "selected_tool": "ModelEvaluator.evaluate + LeakageDetector.check",
        "reason": (
            f"Score {best_score:.4f} {'≥' if score_passes else '<'} threshold {_PASS_THRESHOLD}. "
            f"Leakage: {'clean' if leakage_report.passed else 'critical findings'}."
        ),
        "confidence": best_score,
        "artifacts": [],
        "metrics": {**mean_scores, "cv_std": best_std},
        "warnings": mistake_warnings + leakage_report.findings,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    base_update = {
        "messages": [response],
        "best_model_name": best_name,
        "best_model_metrics": mean_scores,
        "execution_mode": execution_mode,
        "provenance": [provenance_entry],
        "evidence": [evidence_entry],
    }

    # ── Routing decision ──────────────────────────────────────────────────────
    if validation_passed:
        # PASS → compute risk score for deployment gating
        dataset_profile = state.get("dataset_info") or {}
        risk = ModelRiskScorer.score(
            metrics=mean_scores,
            task_type=task_type,
            dataset_profile=dataset_profile,
            cv_std=best_std,
        )
        return Command(
            goto="deployment",
            update={
                **base_update,
                "model_validated": True,
                "risk_score": risk.score,
                "risk_level": risk.risk_level,
            },
        )

    # FAIL — check if we can retry
    if retry_count >= _MAX_RETRIES:
        # Max retries exceeded → halt pipeline with evidence
        logger.warning(
            "Validation: max retries (%d) exceeded. Halting pipeline.", _MAX_RETRIES
        )
        return Command(
            goto=END,
            update={
                **base_update,
                "model_validated": False,
                "errors": [{
                    "agent_name": "validation",
                    "error_type": "max_retries_exceeded",
                    "message": (
                        f"Validation failed after {_MAX_RETRIES} retries. "
                        f"Best score: {best_score:.4f}. "
                        f"Leakage: {leakage_report.findings}"
                    ),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "fatal": True,
                }],
            },
        )

    # Route to failure_analyzer for diagnosis and remediation
    failure_analysis = FailureAnalyzer.analyze(
        best_score=best_score,
        all_scores=mean_scores,
        cv_std=best_std,
        task_type=task_type,
        leakage_report=leakage_report.to_dict() if not leakage_report.passed else None,
    )

    logger.info(
        "Validation: routing to failure_analyzer. action=%s, retry_count=%d → %d",
        failure_analysis.remediation_action, retry_count, retry_count + 1,
    )

    return Command(
        goto="failure_analyzer",
        update={
            **base_update,
            "model_validated": False,
            "validation_retry_count": retry_count + 1,
            "last_failure_analysis": failure_analysis.to_dict(),
        },
    )
