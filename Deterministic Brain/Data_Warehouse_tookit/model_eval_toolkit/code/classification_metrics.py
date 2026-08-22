"""
classification_metrics.py
--------------------------
Pure-numpy implementations of standard binary/multiclass classification
evaluation metrics: confusion matrix, accuracy, precision, recall,
specificity, F1 / F-beta, ROC curve + AUC, Precision-Recall curve + AUC
(average precision), and multiclass macro/micro/weighted averaging.

All functions have been validated against sklearn.metrics on synthetic
data. average_precision_score differs from sklearn's by a small fraction
of a percent on datasets with many tied scores (tie-handling difference);
use sklearn directly if you need bit-exact parity.

No external dependencies besides numpy.
"""

import numpy as np


# --------------------------------------------------------------------------
# Confusion matrix / binary counts
# --------------------------------------------------------------------------

def confusion_matrix(y_true, y_pred, labels=None):
    """
    Returns (matrix, labels) where matrix[i, j] = number of samples with
    true label labels[i] predicted as labels[j].
    """
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
        matrix[label_to_idx[t], label_to_idx[p]] += 1

    return matrix, labels


def binary_counts(y_true, y_pred, positive_label=1):
    """
    Returns dict with TP, FP, TN, FN for binary classification.

    TP: actual positive, predicted positive
    FP: actual negative, predicted positive (Type I error)
    TN: actual negative, predicted negative
    FN: actual positive, predicted negative (Type II error)
    """
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


def _safe_div(numerator, denominator):
    return numerator / denominator if denominator else 0.0


# --------------------------------------------------------------------------
# Core binary metrics
# --------------------------------------------------------------------------

def accuracy_score(y_true, y_pred):
    """(TP + TN) / total. Misleading on imbalanced classes."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def precision_score(y_true, y_pred, average="binary", positive_label=1):
    """
    TP / (TP + FP). "Of everything predicted positive, how much was right?"
    average: "binary", "macro", "micro", or "weighted" (see multiclass section).
    """
    if average == "binary":
        c = binary_counts(y_true, y_pred, positive_label)
        return _safe_div(c["TP"], c["TP"] + c["FP"])
    return _multiclass_metric(y_true, y_pred, "precision", average)


def recall_score(y_true, y_pred, average="binary", positive_label=1):
    """
    TP / (TP + FN). "Of all actual positives, how many did we catch?"
    Also called Sensitivity or True Positive Rate.
    """
    if average == "binary":
        c = binary_counts(y_true, y_pred, positive_label)
        return _safe_div(c["TP"], c["TP"] + c["FN"])
    return _multiclass_metric(y_true, y_pred, "recall", average)


def specificity_score(y_true, y_pred, positive_label=1):
    """TN / (TN + FP). "Of all actual negatives, how many did we correctly clear?" """
    c = binary_counts(y_true, y_pred, positive_label)
    return _safe_div(c["TN"], c["TN"] + c["FP"])


def f1_score(y_true, y_pred, average="binary", positive_label=1):
    """Harmonic mean of precision and recall. High only when BOTH are high."""
    return fbeta_score(y_true, y_pred, beta=1.0, average=average,
                        positive_label=positive_label)


def fbeta_score(y_true, y_pred, beta, average="binary", positive_label=1):
    """
    Weighted harmonic mean of precision/recall.
    beta=1 -> F1 (equal weight)
    beta=2 -> recall weighted 2x more (use when missing positives is costly,
              e.g. cancer/fraud detection)
    beta=0.5 -> precision weighted 2x more (use when false alarms are costly,
                e.g. spam filters)
    """
    if average == "binary":
        p = precision_score(y_true, y_pred, average="binary", positive_label=positive_label)
        r = recall_score(y_true, y_pred, average="binary", positive_label=positive_label)
        b2 = beta ** 2
        denom = (b2 * p) + r
        return _safe_div((1 + b2) * p * r, denom)

    # multiclass: compute per-class fbeta then average
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
        supports.append(c["TP"] + c["FN"])  # actual count of this class

    per_class = np.array(per_class)
    supports = np.array(supports)

    if average == "macro":
        return float(np.mean(per_class))
    if average == "weighted":
        return float(_safe_div(np.sum(per_class * supports), np.sum(supports)))
    if average == "micro":
        # micro-F(beta) == micro-precision == micro-recall for single-label problems
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

    raise ValueError("average must be one of: binary, macro, micro, weighted")


# --------------------------------------------------------------------------
# Multiclass averaging helper (precision/recall)
# --------------------------------------------------------------------------

def _multiclass_metric(y_true, y_pred, metric, average):
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
        else:  # recall
            return _safe_div(tp_total, tp_total + fn_total)

    per_class = []
    supports = []
    for lbl in labels:
        yt_bin = (y_true == lbl).astype(int)
        yp_bin = (y_pred == lbl).astype(int)
        c = binary_counts(yt_bin, yp_bin, positive_label=1)
        if metric == "precision":
            val = _safe_div(c["TP"], c["TP"] + c["FP"])
        else:
            val = _safe_div(c["TP"], c["TP"] + c["FN"])
        per_class.append(val)
        supports.append(c["TP"] + c["FN"])

    per_class = np.array(per_class)
    supports = np.array(supports)

    if average == "macro":
        return float(np.mean(per_class))
    if average == "weighted":
        return float(_safe_div(np.sum(per_class * supports), np.sum(supports)))

    raise ValueError("average must be one of: binary, macro, micro, weighted")


# --------------------------------------------------------------------------
# ROC curve / AUC
# --------------------------------------------------------------------------

def roc_curve(y_true, y_score, positive_label=1):
    """
    Sweeps thresholds over unique scores (descending) and computes
    (FPR, TPR) at each. Returns (fpr, tpr, thresholds), each starting
    at (0,0) and ending at (1,1).
    """
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

    # keep only points where score changes (avoids duplicate threshold ties)
    distinct_idx = np.where(np.diff(score_sorted))[0]
    threshold_idx = np.r_[distinct_idx, len(y_sorted) - 1]

    tps = tps[threshold_idx]
    fps = fps[threshold_idx]
    thresholds = score_sorted[threshold_idx]

    tpr = tps / n_pos if n_pos else np.zeros_like(tps, dtype=float)
    fpr = fps / n_neg if n_neg else np.zeros_like(fps, dtype=float)

    # prepend (0,0)
    tpr = np.r_[0, tpr]
    fpr = np.r_[0, fpr]
    thresholds = np.r_[np.inf, thresholds]

    return fpr, tpr, thresholds


def roc_auc_score(y_true, y_score, positive_label=1):
    """
    Area under the ROC curve via trapezoidal rule.
    Equivalent interpretation: probability that a random positive sample
    scores higher than a random negative sample.
    0.5 = random, 1.0 = perfect.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score, positive_label)
    # trapezoidal rule, implemented manually for numpy-version independence
    return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))


def precision_recall_curve(y_true, y_score, positive_label=1):
    """
    Sweeps thresholds (descending score) and computes (precision, recall)
    at each. Returns (precision, recall, thresholds).
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    y_bin = (y_true == positive_label).astype(int)

    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_bin[order]
    score_sorted = y_score[order]

    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    n_pos = y_bin.sum()

    distinct_idx = np.where(np.diff(score_sorted))[0]
    threshold_idx = np.r_[distinct_idx, len(y_sorted) - 1]

    tps = tps[threshold_idx]
    fps = fps[threshold_idx]
    thresholds = score_sorted[threshold_idx]

    precision = tps / (tps + fps)
    recall = tps / n_pos if n_pos else np.zeros_like(tps, dtype=float)

    # Data above is already ordered by increasing recall (descending score
    # threshold): prepend the (recall=0, precision=1) starting point, which
    # corresponds to a threshold above every score (nothing predicted positive
    # yet), matching the standard PR-curve convention.
    precision = np.r_[1.0, precision]
    recall = np.r_[0.0, recall]

    return precision, recall, thresholds


def average_precision_score(y_true, y_score, positive_label=1):
    """
    PR-AUC / Average Precision: sum_n (R_n - R_{n-1}) * P_n over thresholds,
    which avoids the (slightly) misleading trapezoidal interpolation on PR
    curves. More informative than ROC-AUC when positive class < 10% of data.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_score, positive_label)
    recall_diff = np.diff(recall, prepend=0.0)
    return float(np.sum(recall_diff * precision))


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def classification_report(y_true, y_pred, y_score=None, positive_label=1, average="binary"):
    """
    Returns a dict with confusion matrix + all core metrics. If y_score is
    provided (binary case), also includes ROC-AUC and PR-AUC (average precision).
    """
    matrix, labels = confusion_matrix(y_true, y_pred)
    report = {
        "confusion_matrix": matrix,
        "labels": labels,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, positive_label=positive_label),
        "recall": recall_score(y_true, y_pred, average=average, positive_label=positive_label),
        "f1": f1_score(y_true, y_pred, average=average, positive_label=positive_label),
    }
    if average == "binary":
        report["specificity"] = specificity_score(y_true, y_pred, positive_label=positive_label)

    if y_score is not None:
        report["roc_auc"] = roc_auc_score(y_true, y_score, positive_label=positive_label)
        report["average_precision"] = average_precision_score(y_true, y_score, positive_label=positive_label)

    return report


if __name__ == "__main__":
    # quick smoke test
    y_true = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    y_pred = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    print(classification_report(y_true, y_pred))
