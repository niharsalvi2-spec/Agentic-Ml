"""
Metrics metadata, task-aware MetricRegistry, and evaluation direction definitions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)

try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    import numpy as np
    def root_mean_squared_error(y_true, y_pred, **kwargs):  # type: ignore[misc]
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

try:
    from sklearn.metrics import davies_bouldin_score as davies_bouldin_index
except ImportError:
    def davies_bouldin_index(X, labels):  # type: ignore[misc]
        from sklearn.metrics import davies_bouldin_score
        return davies_bouldin_score(X, labels)


METRIC_DIRECTION: Dict[str, str] = {
    "accuracy": "maximize",
    "precision": "maximize",
    "recall": "maximize",
    "f1": "maximize",
    "roc_auc": "maximize",
    "pr_auc": "maximize",
    "r2": "maximize",
    "adjusted_r2": "maximize",
    "mae": "minimize",
    "mse": "minimize",
    "rmse": "minimize",
    "loss": "minimize",
    "silhouette": "maximize",
    "davies_bouldin": "minimize",
    "calinski_harabasz": "maximize",
}


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    direction: str                     # "maximize" | "minimize"
    task_compatibility: Tuple[str, ...] # ("classification", "regression", "clustering")
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    threshold_pass: Optional[float] = None

    def is_better(self, val_a: float, val_b: float) -> bool:
        """Return True if val_a is strictly better than val_b."""
        if self.direction == "maximize":
            return val_a > val_b
        return val_a < val_b


class MetricRegistry:
    """Central registry of valid evaluation metrics across all ML tasks."""

    _REGISTRY: Dict[str, MetricDefinition] = {
        "accuracy": MetricDefinition("accuracy", "maximize", ("classification",), 0.0, 1.0, 0.70),
        "precision": MetricDefinition("precision", "maximize", ("classification",), 0.0, 1.0, 0.65),
        "recall": MetricDefinition("recall", "maximize", ("classification",), 0.0, 1.0, 0.65),
        "f1": MetricDefinition("f1", "maximize", ("classification",), 0.0, 1.0, 0.65),
        "roc_auc": MetricDefinition("roc_auc", "maximize", ("classification",), 0.0, 1.0, 0.70),
        "r2": MetricDefinition("r2", "maximize", ("regression",), None, 1.0, 0.50),
        "rmse": MetricDefinition("rmse", "minimize", ("regression",), 0.0, None, 1.0),
        "mae": MetricDefinition("mae", "minimize", ("regression",), 0.0, None, 1.0),
        "mse": MetricDefinition("mse", "minimize", ("regression",), 0.0, None, 1.0),
        "silhouette": MetricDefinition("silhouette", "maximize", ("clustering",), -1.0, 1.0, 0.30),
        "davies_bouldin": MetricDefinition("davies_bouldin", "minimize", ("clustering",), 0.0, None, 1.5),
    }

    @classmethod
    def get(cls, name: str) -> Optional[MetricDefinition]:
        return cls._REGISTRY.get(name.lower())

    @classmethod
    def list_for_task(cls, task_type: str) -> List[MetricDefinition]:
        return [m for m in cls._REGISTRY.values() if task_type in m.task_compatibility]

    @classmethod
    def validate_metric_for_task(cls, name: str, task_type: str) -> bool:
        metric = cls.get(name)
        if not metric:
            return False
        return task_type in metric.task_compatibility


@dataclass
class PrimaryMetric:
    name: str
    value: float
    direction: str  # "maximize" | "minimize"

    @property
    def is_better_higher(self) -> bool:
        return self.direction == "maximize"


def extract_primary_metric(metrics: Dict[str, float], task_type: str = "classification") -> PrimaryMetric:
    """
    Extract the primary benchmark metric for a task type respecting optimization direction.
    """
    if not metrics:
        raise ValueError("Cannot extract primary metric from empty metrics dictionary.")

    if task_type == "classification":
        for preferred in ["f1", "accuracy", "roc_auc", "precision", "recall"]:
            if preferred in metrics:
                return PrimaryMetric(
                    name=preferred,
                    value=float(metrics[preferred]),
                    direction=METRIC_DIRECTION.get(preferred, "maximize"),
                )
    elif task_type == "clustering":
        if "silhouette" in metrics:
            return PrimaryMetric(
                name="silhouette",
                value=float(metrics["silhouette"]),
                direction="maximize",
            )
    else:  # regression
        for preferred in ["r2", "rmse", "mae", "mse"]:
            if preferred in metrics:
                return PrimaryMetric(
                    name=preferred,
                    value=float(metrics[preferred]),
                    direction=METRIC_DIRECTION.get(preferred, "maximize" if preferred == "r2" else "minimize"),
                )

    # Default to first available metric
    first_name, first_val = next(iter(metrics.items()))
    return PrimaryMetric(
        name=first_name,
        value=float(first_val),
        direction=METRIC_DIRECTION.get(first_name.lower(), "maximize"),
    )


def classification_report(
    y_true,
    y_pred,
    y_score=None,
    positive_label: int = 1,
    average: str = "binary",
) -> Dict[str, Any]:
    """Compute classification metrics dict from ground truth and predictions."""
    report: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_score is not None:
        try:
            import numpy as np
            score_arr = np.array(y_score)
            if score_arr.ndim == 2:
                score_arr = score_arr[:, 1]
            report["roc_auc"] = float(roc_auc_score(y_true, score_arr))
        except Exception:
            pass
    return report


def regression_report(y_true, y_pred, n_features: Optional[int] = None) -> Dict[str, Any]:
    """Compute regression metrics dict from ground truth and predictions."""
    import numpy as np

    mse = float(mean_squared_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    report: Dict[str, Any] = {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": r2,
    }
    if n_features is not None:
        n = len(y_true)
        if n > n_features + 1:
            adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
            report["adjusted_r2"] = float(adj_r2)
    return report


def clustering_report(X, labels_pred) -> Dict[str, Any]:
    """Compute clustering metrics dict from features and predicted cluster labels."""
    try:
        from sklearn.metrics import silhouette_score, davies_bouldin_score
        unique_labels = set(labels_pred)
        if len(unique_labels) < 2:
            return {"silhouette": 0.0, "davies_bouldin": float("inf"), "n_clusters": len(unique_labels)}
        sil = float(silhouette_score(X, labels_pred))
        db = float(davies_bouldin_score(X, labels_pred))
        return {"silhouette": sil, "davies_bouldin": db, "n_clusters": len(unique_labels)}
    except Exception as exc:
        return {"silhouette": 0.0, "davies_bouldin": float("inf"), "error": str(exc)}
