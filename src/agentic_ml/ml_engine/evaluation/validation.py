"""
Model Validation and Evaluation Agent Engine.
Provides cross-validation evaluation, metric recommendation, and anti-pattern/leakage detection.
"""

from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

from src.agentic_ml.ml_engine.evaluation.metrics import (
    classification_report,
    regression_report,
    clustering_report
)


class EvaluationAgent:
    """Decision-support and evaluation layer with anti-pattern detection."""

    def recommend_metrics(self, task: str, **flags) -> Dict[str, Any]:
        if task == "classification":
            imbalanced = flags.get("imbalanced", False)
            fn_costly = flags.get("fn_costly", False)
            fp_costly = flags.get("fp_costly", False)
            primary = ["accuracy", "f1"] if not imbalanced else ["f1", "recall" if fn_costly else "precision"]
            return {
                "primary": primary,
                "also_report": ["confusion_matrix", "roc_auc"],
                "reasoning": "Standard balanced metrics" if not imbalanced else "Imbalanced class metric focus"
            }
        elif task == "clustering":
            return {
                "primary": ["silhouette", "davies_bouldin"],
                "also_report": ["inertia"],
                "reasoning": "Unsupervised internal cluster validation"
            }
        else:  # regression
            return {
                "primary": ["rmse", "mae", "r2"],
                "also_report": ["adjusted_r2"],
                "reasoning": "Standard regression fit quality and error spread"
            }

    def evaluate_classification(self, y_true, y_pred, y_score=None, positive_label=1, average="binary") -> Dict[str, Any]:
        return classification_report(y_true, y_pred, y_score=y_score, positive_label=positive_label, average=average)

    def evaluate_regression(self, y_true, y_pred, n_features: Optional[int] = None) -> Dict[str, Any]:
        return regression_report(y_true, y_pred, n_features=n_features)

    def evaluate_clustering(self, X, labels_pred) -> Dict[str, Any]:
        return clustering_report(X, labels_pred)

    def check_common_mistakes(self, task: str, **context) -> List[str]:
        """Detects 7 classic evaluation anti-patterns and leakage signals."""
        warnings = []

        if context.get("evaluated_on_training_data"):
            warnings.append("Mistake: Model evaluated on training data rather than a held-out test split.")

        if task == "classification":
            balance = context.get("class_balance")
            if context.get("used_only_accuracy") and balance is not None and balance > 0.8:
                warnings.append(
                    f"Mistake: Relying on accuracy alone with class imbalance ({balance:.0%} majority). "
                    "Report precision/recall/F1 alongside accuracy."
                )
            if context.get("threshold_tuned_on_test_set"):
                warnings.append("Mistake: Tuning decision threshold on the test set causes evaluation leakage.")

        if context.get("scaler_fit_on") == "train+test":
            warnings.append("Mistake: Data leakage detected. Scalers/encoders were fit on full dataset instead of train split.")

        train_score = context.get("train_score")
        val_score = context.get("val_score")
        if train_score is not None and val_score is not None:
            if (train_score - val_score) > 0.15:
                warnings.append(
                    f"Overfitting detected: Train score ({train_score:.3f}) exceeds Val score ({val_score:.3f}) by >0.15."
                )

        return warnings


class ModelEvaluator:
    """Evaluates and cross-validates candidate models."""

    @staticmethod
    def evaluate(
        models: Dict[str, Any],
        X: pd.DataFrame,
        y: pd.Series,
        task_type: str = "classification",
        cv: int = 3
    ) -> Tuple[str, Dict[str, float]]:
        best_model_name = ""
        best_score = -float("inf")
        results = {}
        
        scoring = "accuracy" if task_type == "classification" else "r2"
        
        for name, model in models.items():
            try:
                scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
                mean_score = float(scores.mean())
                results[name] = round(mean_score, 4)
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_model_name = name
            except Exception:
                results[name] = 0.0

        return best_model_name, results
