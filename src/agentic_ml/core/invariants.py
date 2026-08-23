"""
Centralized invariant verification system.

Ensures that no agent can proceed with missing data, unvalidated models,
fabricated fallbacks, or unapproved deployments.
"""
from typing import Dict, Any, List, Optional


class InvariantViolation(RuntimeError):
    """Raised when a core pipeline state invariant is violated."""
    pass


def require_raw_data(state: Dict[str, Any]) -> None:
    """Ensure data collection succeeded and produced a valid dataframe or dataset path."""
    if not state.get("data_collected"):
        raise InvariantViolation("Invariant violation: data_collected must be True.")
    if state.get("raw_df") is None and not state.get("dataset_path"):
        raise InvariantViolation("Invariant violation: raw_df or dataset_path required.")


def require_preprocessed_features(state: Dict[str, Any]) -> None:
    """Ensure preprocessing succeeded with feature matrix X and target y."""
    if not state.get("data_preprocessed"):
        raise InvariantViolation("Invariant violation: data_preprocessed must be True.")
    if state.get("X") is None or state.get("y") is None:
        raise InvariantViolation("Invariant violation: feature matrix X and target y required.")


def require_trained_models(state: Dict[str, Any]) -> None:
    """Ensure model building produced at least one candidate model."""
    if not state.get("model_built"):
        raise InvariantViolation("Invariant violation: model_built must be True.")
    trained = state.get("trained_models") or {}
    if not trained:
        raise InvariantViolation("Invariant violation: trained_models cannot be empty.")


def require_tested_models(state: Dict[str, Any]) -> None:
    """Ensure model testing passed for at least one candidate."""
    if not state.get("model_tested"):
        raise InvariantViolation("Invariant violation: model_tested must be True.")


def require_validated_model(state: Dict[str, Any]) -> str:
    """Ensure validation crowned a winning model with metric evidence."""
    if not state.get("model_validated"):
        raise InvariantViolation("Invariant violation: model_validated must be True.")
    best_name = state.get("best_model_name")
    if not best_name:
        raise InvariantViolation("Invariant violation: best_model_name cannot be None or empty.")
    metrics = state.get("best_model_metrics") or {}
    if not metrics:
        raise InvariantViolation("Invariant violation: best_model_metrics cannot be empty.")
    return best_name


def require_deployment_approval(state: Dict[str, Any]) -> str:
    """Ensure deployment has explicit approval (AUTO_APPROVE or HUMAN_APPROVED)."""
    decision = state.get("deployment_decision")
    if decision not in {"AUTO_APPROVE", "HUMAN_APPROVED"}:
        raise InvariantViolation(
            f"Invariant violation: Deployment requires explicit approval; got {decision!r}."
        )
    return decision
