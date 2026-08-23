"""
Truthful completion gate for pipeline execution.

Validates that a pipeline run truly satisfied all required stage invariants
before declaring RUN_COMPLETED.
"""
from typing import Dict, Any, Tuple, List


def verify_run_completion(state: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Verify truthful pipeline completion against evidence invariants.

    Returns:
        (passed, list_of_failure_reasons)
    """
    failures: List[str] = []

    if not state.get("problem_analyzed"):
        failures.append("problem_analyzed is False")

    if not state.get("data_collected"):
        failures.append("data_collected is False")

    if not state.get("data_preprocessed"):
        failures.append("data_preprocessed is False")

    if not state.get("model_built"):
        failures.append("model_built is False")

    if not state.get("model_tested"):
        failures.append("model_tested is False")

    if not state.get("model_validated"):
        failures.append("model_validated is False")

    if not state.get("deployment_completed"):
        failures.append("deployment_completed is False")

    if not state.get("best_model_name"):
        failures.append("best_model_name is missing")

    if not state.get("best_model_metrics"):
        failures.append("best_model_metrics is missing")

    if not state.get("artifact_path"):
        failures.append("artifact_path is missing")

    return (len(failures) == 0, failures)
