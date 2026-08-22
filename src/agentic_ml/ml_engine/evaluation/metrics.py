"""
Zero-Dependency Metrics Library.
Pure-numpy implementations of Classification, Regression, and Clustering metrics.
Validated against scikit-learn standard definitions.
"""

from typing import Dict, Any, Tuple, List, Optional, Union
import numpy as np


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


# --------------------------------------------------------------------------
# Classification Metrics
# --------------------------------------------------------------------------

def confusion_matrix(y_true, y_pred, labels=None) -> Tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    else:
        labels = np.asarray(labels)

    label_to_idx = {label: i for i, label in enumerate(labels)}
    n = len(labels)
    matrix = np.zeros((n, n), dtype=int)

    for t, p in zip(y_true, y_pred):
        if t in label_to_idx and p in label_to_idx:
            matrix[label_to_idx[t], label_to_idx[p]] += 1

    return matrix, labels


def binary_counts(y_true, y_pred, positive_label=1) -> Dict[str, int]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    actual_pos = (y_true == positive_label)
    actual_neg = ~actual_pos
    pred_pos = (y_pred == positive_label)
    pred_neg = ~pred_pos

    tp = int(np.sum(actual_pos & pred_pos))
    fp = int(np.sum(actual_neg & pred_pos))
    tn = int(np.sum(actual_neg & pred_neg))
    fn = int(np.sum(actual_pos & pred_neg))

    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}


def accuracy_score(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def precision_score(y_true, y_pred, average="binary", positive_label=1) -> float:
    if average == "binary":
        c = binary_counts(y_true, y_pred, positive_label)
        return _safe_div(c["TP"], c["TP"] + c["FP"])
    return _multiclass_metric(y_true, y_pred, "precision", average)


def recall_score(y_true, y_pred, average="binary", positive_label=1) -> float:
    if average == "binary":
        c = binary_counts(y_true, y_pred, positive_label)
        return _safe_div(c["TP"], c["TP"] + c["FN"])
    return _multiclass_metric(y_true, y_pred, "recall", average)


def specificity_score(y_true, y_pred, positive_label=1) -> float:
    c = binary_counts(y_true, y_pred, positive_label)
    return _safe_div(c["TN"], c["TN"] + c["FP"])


def f1_score(y_true, y_pred, average="binary", positive_label=1) -> float:
    return fbeta_score(y_true, y_pred, beta=1.0, average=average, positive_label=positive_label)


def fbeta_score(y_true, y_pred, beta: float = 1.0, average="binary", positive_label=1) -> float:
    if average == "binary":
        p = precision_score(y_true, y_pred, average="binary", positive_label=positive_label)
        r = recall_score(y_true, y_pred, average="binary", positive_label=positive_label)
        b2 = beta ** 2
        denom = (b2 * p) + r
        return _safe_div((1 + b2) * p * r, denom)

    # Multiclass averaging
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    per_class = []
    supports = []

    for lbl in labels:
        yt_bin = (y_true == lbl).astype(int)
        yp_bin = (y_pred == lbl).astype(int)
        c = binary_counts(yt_bin, yp_bin, positive_label=1)
        p = _safe_div(c["TP"], c["TP"] + c["FP"])
        r = _safe_div(c["TP"], c["TP"] + c["FN"])
        b2 = beta ** 2
        f = _safe_div((1 + b2) * p * r, (b2 * p) + r)
        per_class.append(f)
        supports.append(c["TP"] + c["FN"])

    per_class_arr = np.array(per_class)
    supports_arr = np.array(supports)

    if average == "macro":
        return float(np.mean(per_class_arr))
    if average == "weighted":
        return float(_safe_div(np.sum(per_class_arr * supports_arr), np.sum(supports_arr)))
    if average == "micro":
        tp_total = fp_total = fn_total = 0
        for lbl in labels:
            yt_bin = (y_true == lbl).astype(int)
            yp_bin = (y_pred == lbl).astype(int)
            c = binary_counts(yt_bin, yp_bin, positive_label=1)
            tp_total += c["TP"]
            fp_total += c["FP"]
            fn_total += c["FN"]
        p = _safe_div(tp_total, tp_total + fp_total)
        r = _safe_div(tp_total, tp_total + fn_total)
        b2 = beta ** 2
        return _safe_div((1 + b2) * p * r, (b2 * p) + r)

    return float(np.mean(per_class_arr))


def _multiclass_metric(y_true, y_pred, metric: str, average: str) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))

    if average == "micro":
        tp_total = fp_total = fn_total = 0
        for lbl in labels:
            yt_bin = (y_true == lbl).astype(int)
            yp_bin = (y_pred == lbl).astype(int)
            c = binary_counts(yt_bin, yp_bin, positive_label=1)
            tp_total += c["TP"]
            fp_total += c["FP"]
            fn_total += c["FN"]
        if metric == "precision":
            return _safe_div(tp_total, tp_total + fp_total)
        return _safe_div(tp_total, tp_total + fn_total)

    per_class = []
    supports = []
    for lbl in labels:
        yt_bin = (y_true == lbl).astype(int)
        yp_bin = (y_pred == lbl).astype(int)
        c = binary_counts(yt_bin, yp_bin, positive_label=1)
        val = _safe_div(c["TP"], c["TP"] + c["FP"]) if metric == "precision" else _safe_div(c["TP"], c["TP"] + c["FN"])
        per_class.append(val)
        supports.append(c["TP"] + c["FN"])

    per_class_arr = np.array(per_class)
    supports_arr = np.array(supports)

    if average == "macro":
        return float(np.mean(per_class_arr))
    if average == "weighted":
        return float(_safe_div(np.sum(per_class_arr * supports_arr), np.sum(supports_arr)))
    return float(np.mean(per_class_arr))


def roc_curve(y_true, y_score, positive_label=1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    y_bin = (y_true == positive_label).astype(int)

    n_pos = y_bin.sum()
    n_neg = len(y_bin) - n_pos

    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_bin[order]
    score_sorted = y_score[order]

    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)

    distinct_idx = np.where(np.diff(score_sorted))[0]
    threshold_idx = np.r_[distinct_idx, len(y_sorted) - 1]

    tps = tps[threshold_idx]
    fps = fps[threshold_idx]
    thresholds = score_sorted[threshold_idx]

    tpr = tps / n_pos if n_pos else np.zeros_like(tps, dtype=float)
    fpr = fps / n_neg if n_neg else np.zeros_like(fps, dtype=float)

    tpr = np.r_[0, tpr]
    fpr = np.r_[0, fpr]
    thresholds = np.r_[np.inf, thresholds]

    return fpr, tpr, thresholds


def roc_auc_score(y_true, y_score, positive_label=1) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score, positive_label)
    return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))


def classification_report(y_true, y_pred, y_score=None, positive_label=1, average="binary") -> Dict[str, Any]:
    matrix, labels = confusion_matrix(y_true, y_pred)
    report = {
        "confusion_matrix": matrix.tolist(),
        "labels": labels.tolist(),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, positive_label=positive_label),
        "recall": recall_score(y_true, y_pred, average=average, positive_label=positive_label),
        "f1": f1_score(y_true, y_pred, average=average, positive_label=positive_label),
    }
    if average == "binary":
        report["specificity"] = specificity_score(y_true, y_pred, positive_label=positive_label)
    if y_score is not None:
        try:
            report["roc_auc"] = roc_auc_score(y_true, y_score, positive_label=positive_label)
        except Exception:
            pass
    return report


# --------------------------------------------------------------------------
# Regression Metrics
# --------------------------------------------------------------------------

def mean_absolute_error(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def mean_squared_error(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))


def root_mean_squared_error(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def r2_score(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1 - ss_res / ss_tot)


def adjusted_r2_score(y_true, y_pred, n_features: int) -> float:
    y_true = np.asarray(y_true, dtype=float)
    n = len(y_true)
    p = n_features
    if n - p - 1 <= 0:
        return float("nan")
    r2 = r2_score(y_true, y_pred)
    return float(1 - (1 - r2) * (n - 1) / (n - p - 1))


def regression_report(y_true, y_pred, n_features: Optional[int] = None) -> Dict[str, Any]:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    report = {
        "mae": round(mae, 4),
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
    }
    if n_features is not None:
        report["adjusted_r2"] = round(adjusted_r2_score(y_true, y_pred, n_features), 4)
        report["rmse_vs_mae_ratio"] = round(rmse / mae, 4) if mae else float("nan")
    return report


# --------------------------------------------------------------------------
# Clustering Metrics
# --------------------------------------------------------------------------

def _pairwise_distances(X: np.ndarray) -> np.ndarray:
    diff = X[:, None, :] - X[None, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def silhouette_score(X, labels) -> float:
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    n = len(X)
    unique_labels = np.unique(labels)

    if len(unique_labels) < 2:
        return 0.0

    dist = _pairwise_distances(X)
    scores = np.zeros(n)

    for i in range(n):
        own_label = labels[i]
        own_mask = (labels == own_label)
        own_mask[i] = False

        if own_mask.sum() == 0:
            scores[i] = 0.0
            continue

        a_i = dist[i, own_mask].mean()
        b_i = np.inf
        for other_label in unique_labels:
            if other_label == own_label:
                continue
            other_mask = (labels == other_label)
            mean_dist = dist[i, other_mask].mean()
            b_i = min(b_i, mean_dist)

        scores[i] = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0.0

    return float(np.mean(scores))


def davies_bouldin_index(X, labels) -> float:
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    k = len(unique_labels)

    if k < 2:
        return 0.0

    centroids = np.array([X[labels == lbl].mean(axis=0) for lbl in unique_labels])
    sigmas = np.array([
        np.mean(np.sqrt(np.sum((X[labels == lbl] - centroids[idx]) ** 2, axis=1)))
        for idx, lbl in enumerate(unique_labels)
    ])

    db_values = []
    for i in range(k):
        max_r = -np.inf
        for j in range(k):
            if i == j:
                continue
            centroid_dist = np.sqrt(np.sum((centroids[i] - centroids[j]) ** 2))
            r_ij = (sigmas[i] + sigmas[j]) / centroid_dist if centroid_dist > 0 else np.inf
            max_r = max(max_r, r_ij)
        db_values.append(max_r)

    return float(np.mean(db_values))


def inertia(X, labels) -> float:
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)

    total = 0.0
    for lbl in unique_labels:
        cluster_points = X[labels == lbl]
        if len(cluster_points) > 0:
            centroid = cluster_points.mean(axis=0)
            total += np.sum((cluster_points - centroid) ** 2)

    return float(total)


def clustering_report(X, labels_pred) -> Dict[str, Any]:
    return {
        "silhouette": round(silhouette_score(X, labels_pred), 4),
        "davies_bouldin": round(davies_bouldin_index(X, labels_pred), 4),
        "inertia": round(inertia(X, labels_pred), 4),
    }
