"""
Model Registry and Recommendation Engine.
Contains metadata definitions for all models and intelligent rule-based recommenders.
"""

from typing import Dict, Any, List, Optional
from src.agentic_ml.ml_engine.models.classification import get_classification_models
from src.agentic_ml.ml_engine.models.regression import get_regression_models
from src.agentic_ml.ml_engine.models.clustering import get_clustering_models

CLASSIFIER_METADATA: Dict[str, Dict[str, Any]] = {
    "LogisticRegression": {
        "interpretable": True, "supports_proba": True, "handles_nonlinear": False,
        "sensitive_to_outliers": True, "handles_imbalance_well": True,
        "inference_speed": "fast", "good_for_high_dim": True, "good_for_small_data": True,
        "good_for_large_data": True
    },
    "NaiveBayes": {
        "interpretable": True, "supports_proba": True, "handles_nonlinear": False,
        "sensitive_to_outliers": False, "handles_imbalance_well": True,
        "inference_speed": "fast", "good_for_high_dim": True, "good_for_small_data": True,
        "good_for_large_data": True
    },
    "KNN": {
        "interpretable": False, "supports_proba": True, "handles_nonlinear": True,
        "sensitive_to_outliers": True, "handles_imbalance_well": False,
        "inference_speed": "slow", "good_for_high_dim": False, "good_for_small_data": True,
        "good_for_large_data": False
    },
    "DecisionTree": {
        "interpretable": True, "supports_proba": True, "handles_nonlinear": True,
        "sensitive_to_outliers": False, "handles_imbalance_well": True,
        "inference_speed": "fast", "good_for_high_dim": False, "good_for_small_data": True,
        "good_for_large_data": True
    },
    "RandomForest": {
        "interpretable": False, "supports_proba": True, "handles_nonlinear": True,
        "sensitive_to_outliers": False, "handles_imbalance_well": True,
        "inference_speed": "fast", "good_for_high_dim": True, "good_for_small_data": True,
        "good_for_large_data": True
    },
    "GradientBoosting": {
        "interpretable": False, "supports_proba": True, "handles_nonlinear": True,
        "sensitive_to_outliers": False, "handles_imbalance_well": True,
        "inference_speed": "fast", "good_for_high_dim": True, "good_for_small_data": True,
        "good_for_large_data": True
    },
    "SVM": {
        "interpretable": False, "supports_proba": True, "handles_nonlinear": True,
        "sensitive_to_outliers": True, "handles_imbalance_well": False,
        "inference_speed": "medium", "good_for_high_dim": True, "good_for_small_data": True,
        "good_for_large_data": False
    },
}


class ModelRegistry:
    """Unified access to all model families and rule-based architecture recommendations."""

    @staticmethod
    def get_models_for_task(task_type: str = "classification", random_state: int = 42) -> Dict[str, Any]:
        if task_type == "classification":
            return get_classification_models(random_state)
        elif task_type == "clustering":
            return get_clustering_models(random_state=random_state)
        else:
            return get_regression_models(random_state)

    @staticmethod
    def recommend(
        task_type: str = "classification",
        n_samples: Optional[int] = None,
        n_features: Optional[int] = None,
        need_interpretability: bool = False,
        need_proba: bool = False,
        suspect_nonlinear: bool = False,
        has_outliers: bool = False,
        is_imbalanced: bool = False,
        need_fast_inference: bool = False,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Scores candidate models based on problem constraints and returns top recommendations."""
        if task_type != "classification":
            # Default recommendation for regression
            if need_interpretability:
                return [{"model": "Ridge", "score": 10, "reasons": ["interpretable linear model"]}]
            return [{"model": "GradientBoosting", "score": 10, "reasons": ["high accuracy tree ensemble"]}]

        scores = {}
        reasons = {}
        is_high_dim = (n_features > n_samples) if (n_features and n_samples) else False

        for name, meta in CLASSIFIER_METADATA.items():
            score = 0
            why = []

            if need_interpretability and meta["interpretable"]:
                score += 2
                why.append("interpretable")
            if need_proba and meta["supports_proba"]:
                score += 1
                why.append("supports probabilities")
            if suspect_nonlinear and meta["handles_nonlinear"]:
                score += 2
                why.append("handles nonlinear patterns")
            if not suspect_nonlinear and not meta["handles_nonlinear"]:
                score += 1
                why.append("optimal for linear boundaries")
            if has_outliers and not meta["sensitive_to_outliers"]:
                score += 2
                why.append("robust to outliers")
            if is_imbalanced and meta["handles_imbalance_well"]:
                score += 1
                why.append("handles imbalance well")
            if need_fast_inference and meta["inference_speed"] == "fast":
                score += 2
                why.append("low inference latency")
            if is_high_dim and meta["good_for_high_dim"]:
                score += 2
                why.append("suitable for high-dimensional space")

            scores[name] = score
            reasons[name] = why

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [
            {"model": name, "score": score, "reasons": reasons[name]}
            for name, score in ranked[:top_k]
        ]
