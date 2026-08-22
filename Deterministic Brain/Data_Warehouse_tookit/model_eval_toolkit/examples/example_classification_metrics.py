"""
Example: classification_metrics.py

Run with:  python3 example_classification_metrics.py
(run from inside /examples with /code on the path, or add /code to PYTHONPATH)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

import numpy as np
from classification_metrics import (
    confusion_matrix, binary_counts, accuracy_score, precision_score,
    recall_score, specificity_score, f1_score, fbeta_score,
    roc_auc_score, average_precision_score, classification_report,
)


def imbalanced_disease_example():
    print("=" * 70)
    print("Example 1: the accuracy trap on imbalanced data")
    print("=" * 70)

    # 100 patients: 90 healthy (0), 10 sick (1). Lazy model predicts healthy always.
    y_true = [0] * 90 + [1] * 10
    y_pred = [0] * 100

    print("Accuracy:   ", accuracy_score(y_true, y_pred))
    print("Recall:     ", recall_score(y_true, y_pred))  # catches this failure
    print("Precision:  ", precision_score(y_true, y_pred))
    print("F1:         ", f1_score(y_true, y_pred))
    print("-> 90% accuracy but 0% recall: the model never catches a real case.\n")


def spam_filter_example():
    print("=" * 70)
    print("Example 2: precision vs. recall trade-off (spam filter)")
    print("=" * 70)

    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, 500)          # 1 = spam
    y_score = np.clip(y_true * 0.55 + rng.normal(0, 0.35, 500), 0, 1)

    for threshold in [0.3, 0.5, 0.7]:
        y_pred = (y_score > threshold).astype(int)
        p = precision_score(y_true, y_pred)
        r = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        print(f"threshold={threshold:.1f}  precision={p:.3f}  recall={r:.3f}  f1={f1:.3f}")

    print("-> Raising the threshold trades recall for precision.")
    print("   For spam (false positives costly), prefer higher precision.\n")


def roc_pr_auc_example():
    print("=" * 70)
    print("Example 3: ROC-AUC vs PR-AUC on rare-event (fraud) data")
    print("=" * 70)

    rng = np.random.default_rng(7)
    n = 2000
    y_true = (rng.random(n) < 0.02).astype(int)  # 2% fraud rate
    y_score = np.clip(y_true * 0.7 + rng.normal(0, 0.3, n), 0, 1)

    roc_auc = roc_auc_score(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)
    print(f"Positive rate: {y_true.mean():.1%}")
    print(f"ROC-AUC: {roc_auc:.3f}")
    print(f"PR-AUC:  {pr_auc:.3f}")
    print("-> ROC-AUC can look inflated on rare-event data; PR-AUC is the "
          "more honest signal for how well the positive class is detected.\n")


def multiclass_example():
    print("=" * 70)
    print("Example 4: multiclass averaging (macro / weighted / micro)")
    print("=" * 70)

    y_true = [0, 0, 0, 1, 1, 2, 2, 2, 2, 2]
    y_pred = [0, 0, 1, 1, 1, 2, 2, 2, 0, 2]

    for avg in ["macro", "weighted", "micro"]:
        print(f"{avg:8s} precision={precision_score(y_true, y_pred, average=avg):.3f}  "
              f"recall={recall_score(y_true, y_pred, average=avg):.3f}  "
              f"f1={f1_score(y_true, y_pred, average=avg):.3f}")

    matrix, labels = confusion_matrix(y_true, y_pred)
    print("Confusion matrix (rows=true, cols=predicted), labels =", labels)
    print(matrix, "\n")


def full_report_example():
    print("=" * 70)
    print("Example 5: classification_report() end to end")
    print("=" * 70)

    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 300)
    y_score = np.clip(y_true * 0.6 + rng.normal(0, 0.4, 300), 0, 1)
    y_pred = (y_score > 0.5).astype(int)

    report = classification_report(y_true, y_pred, y_score=y_score)
    for k, v in report.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    imbalanced_disease_example()
    spam_filter_example()
    roc_pr_auc_example()
    multiclass_example()
    full_report_example()
