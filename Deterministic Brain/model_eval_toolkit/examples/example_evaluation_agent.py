"""
Example: evaluation_agent.py

Run with:  python3 example_evaluation_agent.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

import numpy as np
from evaluation_agent import EvaluationAgent


def recommend_classification_example(agent):
    print("=" * 70)
    print("Example 1: recommend_metrics() for a fraud-detection classifier")
    print("=" * 70)

    rec = agent.recommend_metrics(
        "classification", imbalanced=True, positive_rate=0.02, fn_costly=True
    )
    print("Primary metrics:   ", rec["primary"])
    print("Also report:       ", rec["also_report"])
    print("Reasoning:         ", rec["reasoning"])
    print()

    rec2 = agent.recommend_metrics(
        "classification", imbalanced=True, positive_rate=0.4, fp_costly=True
    )
    print("Spam filter case")
    print("Primary metrics:   ", rec2["primary"])
    print("Reasoning:         ", rec2["reasoning"], "\n")


def recommend_regression_example(agent):
    print("=" * 70)
    print("Example 2: recommend_metrics() for regression with outliers")
    print("=" * 70)

    rec = agent.recommend_metrics(
        "regression", outliers_in_target=True, comparing_feature_sets=True
    )
    print("Primary metrics:   ", rec["primary"])
    print("Also report:       ", rec["also_report"])
    print("Reasoning:         ", rec["reasoning"], "\n")


def recommend_clustering_example(agent):
    print("=" * 70)
    print("Example 3: recommend_metrics() for choosing K in K-Means")
    print("=" * 70)

    rec = agent.recommend_metrics("clustering", choosing_k=True)
    print("Primary metrics:   ", rec["primary"])
    print("Reasoning:         ", rec["reasoning"], "\n")


def evaluate_end_to_end_example(agent):
    print("=" * 70)
    print("Example 4: evaluate_classification() end to end")
    print("=" * 70)

    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 300)
    y_score = np.clip(y_true * 0.6 + rng.normal(0, 0.4, 300), 0, 1)
    y_pred = (y_score > 0.5).astype(int)

    report = agent.evaluate_classification(y_true, y_pred, y_score=y_score)
    for k, v in report.items():
        print(f"{k}: {v}")
    print()


def check_mistakes_example(agent):
    print("=" * 70)
    print("Example 5: check_common_mistakes() catching a bad evaluation setup")
    print("=" * 70)

    warnings = agent.check_common_mistakes(
        "classification",
        used_only_accuracy=True,
        class_balance=0.95,
        threshold_tuned_on_test_set=True,
        scaler_fit_on="train+test",
        train_score=0.99,
        val_score=0.62,
        metrics_reported=["accuracy"],
    )
    for w in warnings:
        print("-", w)
    print()

    print("A clean setup produces no warnings:")
    clean = agent.check_common_mistakes(
        "classification",
        evaluated_on_training_data=False,
        used_only_accuracy=False,
        threshold_tuned_on_test_set=False,
        scaler_fit_on="train",
        train_score=0.88,
        val_score=0.85,
        metrics_reported=["precision", "recall", "f1", "roc_auc"],
    )
    print(clean if clean else "(no warnings)")


if __name__ == "__main__":
    agent = EvaluationAgent()
    recommend_classification_example(agent)
    recommend_regression_example(agent)
    recommend_clustering_example(agent)
    evaluate_end_to_end_example(agent)
    check_mistakes_example(agent)
