"""
Metrics metadata and evaluation direction definitions.
"""
from dataclasses import dataclass
from typing import Dict, Any, Optional

# ── sklearn re-exports for backward compatibility ──────────────────────────────
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
    "r2": "maximize",
    "mae": "minimize",
    "mse": "minimize",
    "rmse": "minimize",
    "loss": "minimize",
    "silhouette": "maximize",
}


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
        return PrimaryMetric(name="unknown", value=0.0, direction="maximize")

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


# ── Backward-compatible report functions (used by validation.py) ──────────────

def classification_report(
    y_true,
    y_pred,
    y_score=None,
    positive_label: int = 1,
    average: str = "binary",
) -> Dict[str, Any]:
    """Compute classification metrics dict from ground truth and predictions."""
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        confusion_matrix as _cm,
    )
    report: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "confusion_matrix": _cm(y_true, y_pred).tolist(),
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
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
