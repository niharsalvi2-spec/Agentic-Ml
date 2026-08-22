"""
evaluation_agent.py
---------------------
EvaluationAgent: a small decision-support layer on top of
classification_metrics / regression_metrics / clustering_metrics.

It does two things a metrics library on its own doesn't:
  1. recommend_metrics()   -- tells you WHICH metrics fit your task, before
                               you compute anything, based on task type and
                               a few situational flags (imbalance, outliers,
                               cost asymmetry, ground truth availability...).
  2. evaluate_*()          -- runs the recommended metrics and returns a
                               single report dict.
  3. check_common_mistakes() -- flags the 7 classic evaluation mistakes
                               (train/test leakage signals, accuracy on
                               imbalanced data, threshold tuned on test set,
                               single-metric reporting, etc.) given whatever
                               context you can supply.
"""

import numpy as np

from classification_metrics import classification_report
from regression_metrics import regression_report
from clustering_metrics import clustering_report


class EvaluationAgent:

    # ----------------------------------------------------------------
    # 1. Metric recommendation (decision guide, made programmatic)
    # ----------------------------------------------------------------

    def recommend_metrics(self, task, **flags):
        """
        task: "classification" | "regression" | "clustering"

        Relevant flags:
          classification: imbalanced (bool), positive_rate (float, 0-1),
                           fn_costly (bool), fp_costly (bool)
          regression:      outliers_in_target (bool), large_errors_worse (bool),
                            comparing_feature_sets (bool)
          clustering:       has_ground_truth (bool), choosing_k (bool),
                            comparing_algorithms (bool)

        Returns {"primary": [...], "also_report": [...], "reasoning": str}
        """
        if task == "classification":
            return self._recommend_classification(**flags)
        if task == "regression":
            return self._recommend_regression(**flags)
        if task == "clustering":
            return self._recommend_clustering(**flags)
        raise ValueError("task must be 'classification', 'regression', or 'clustering'")

    def _recommend_classification(self, imbalanced=False, positive_rate=None,
                                   fn_costly=False, fp_costly=False, **_):
        primary = []
        reasoning = []

        rare_positive = (positive_rate is not None and positive_rate < 0.10)

        if not imbalanced:
            primary = ["accuracy", "f1"]
            reasoning.append("Classes are balanced: accuracy is meaningful, "
                              "F1 balances precision/recall.")
        else:
            reasoning.append("Classes are imbalanced: accuracy alone is misleading "
                              "(a majority-class model can score high while being useless).")
            if fn_costly and not fp_costly:
                primary.append("recall")
                reasoning.append("Missing positives is costly -> prioritize recall.")
            elif fp_costly and not fn_costly:
                primary.append("precision")
                reasoning.append("False alarms are costly -> prioritize precision.")
            else:
                primary.append("f1_or_fbeta")
                reasoning.append("Both error types matter -> use F1 or a weighted F-beta.")

            if rare_positive:
                primary.append("pr_auc")
                reasoning.append("Positive class < 10% of data -> prefer PR-AUC over ROC-AUC.")
            else:
                primary.append("roc_auc")

        also_report = ["confusion_matrix"]
        if fn_costly:
            also_report.append("fbeta(beta=2)")
        if fp_costly:
            also_report.append("fbeta(beta=0.5)")

        return {"primary": primary, "also_report": also_report,
                "reasoning": " ".join(reasoning)}

    def _recommend_regression(self, outliers_in_target=False, large_errors_worse=False,
                               comparing_feature_sets=False, **_):
        primary = []
        reasoning = []

        if outliers_in_target:
            primary.append("mae")
            reasoning.append("Outliers present in target -> MAE is robust and interpretable.")
        if large_errors_worse:
            primary.append("rmse")
            reasoning.append("Large errors should be penalized disproportionately -> RMSE.")
        if comparing_feature_sets:
            primary.append("adjusted_r2")
            reasoning.append("Comparing models with different feature counts -> use Adjusted R2, "
                              "not plain R2 (which never decreases when features are added).")
        if not primary:
            primary = ["mae", "rmse", "r2"]
            reasoning.append("No special constraints -> report MAE, RMSE, and R2 together "
                              "(typical error, outlier sensitivity, and fit quality).")

        also_report = [m for m in ["mae", "rmse", "r2"] if m not in primary]
        return {"primary": primary, "also_report": also_report,
                "reasoning": " ".join(reasoning)}

    def _recommend_clustering(self, has_ground_truth=False, choosing_k=False,
                               comparing_algorithms=False, **_):
        primary = []
        reasoning = []

        if has_ground_truth:
            primary = ["adjusted_rand_index", "nmi"]
            reasoning.append("Ground truth labels available -> use external metrics "
                              "ARI (chance-corrected) and NMI (information-theoretic).")
        else:
            reasoning.append("No ground truth -> use internal metrics based on cluster structure.")
            if choosing_k:
                primary.append("inertia_elbow")
                primary.append("silhouette")
                reasoning.append("Choosing K -> Elbow method on Inertia, pick highest Silhouette.")
            if comparing_algorithms:
                primary.append("silhouette")
                primary.append("davies_bouldin")
                reasoning.append("Comparing algorithms -> Silhouette (higher better) and "
                                  "Davies-Bouldin (lower better).")
            if not primary:
                primary = ["silhouette", "davies_bouldin"]

        return {"primary": list(dict.fromkeys(primary)), "also_report": [],
                "reasoning": " ".join(reasoning)}

    # ----------------------------------------------------------------
    # 2. Run evaluation
    # ----------------------------------------------------------------

    def evaluate_classification(self, y_true, y_pred, y_score=None,
                                 positive_label=1, average="binary"):
        return classification_report(y_true, y_pred, y_score=y_score,
                                       positive_label=positive_label, average=average)

    def evaluate_regression(self, y_true, y_pred, n_features=None):
        return regression_report(y_true, y_pred, n_features=n_features)

    def evaluate_clustering(self, X, labels_pred, labels_true=None):
        return clustering_report(X, labels_pred, labels_true=labels_true)

    # ----------------------------------------------------------------
    # 3. Common mistake checks
    # ----------------------------------------------------------------

    def check_common_mistakes(self, task, **context):
        """
        Checks flags supplied in `context` and returns a list of warning
        strings for any mistake pattern detected. Supply whatever you know;
        unknown flags are simply ignored.

        Recognized context keys:
          evaluated_on_training_data (bool)
          class_balance (float, fraction of majority class) [classification]
          used_only_accuracy (bool)                          [classification]
          threshold_tuned_on_test_set (bool)                 [classification]
          used_equal_error_costs (bool)                      [classification/regression]
          scaler_fit_on (str: "train" | "train+test")
          train_score (float), val_score (float)             [any task]
          metrics_reported (list[str])
        """
        warnings = []

        if context.get("evaluated_on_training_data"):
            warnings.append(
                "Mistake: evaluating on training data. The model has already "
                "memorized it and will look artificially perfect. Always score "
                "on a held-out validation/test set the model never trained on."
            )

        if task == "classification":
            balance = context.get("class_balance")
            if context.get("used_only_accuracy") and balance is not None and balance > 0.8:
                warnings.append(
                    f"Mistake: relying on accuracy alone with a majority class at "
                    f"~{balance:.0%}. A model that always predicts the majority class "
                    "would score high while catching none of the minority class. "
                    "Report precision/recall/F1/PR-AUC alongside it."
                )
            if context.get("threshold_tuned_on_test_set"):
                warnings.append(
                    "Mistake: tuning the decision threshold on the test set. This "
                    "turns the test set into a validation set and inflates the "
                    "reported test performance. Tune on validation, report test once."
                )

        if context.get("used_equal_error_costs") and context.get("costs_actually_asymmetric"):
            warnings.append(
                "Mistake: weighting all errors equally when the real-world costs "
                "of false positives vs. false negatives differ (e.g. a missed "
                "diagnosis vs. a false alarm). Use F-beta or a cost-sensitive metric."
            )

        metrics_reported = context.get("metrics_reported")
        if metrics_reported is not None and len(metrics_reported) <= 1:
            warnings.append(
                "Mistake: reporting a single metric. A high R2/AUC/accuracy can "
                "hide systematic bias or a fragile decision boundary. Always pair "
                "a headline metric with a confusion matrix or residual plot."
            )

        if context.get("scaler_fit_on") == "train+test":
            warnings.append(
                "Mistake: data leakage. Preprocessors (scalers, encoders) were fit "
                "using test-set statistics, which inflates performance. Fit all "
                "preprocessing on the training set only, then transform test data "
                "with those same parameters."
            )

        train_score = context.get("train_score")
        val_score = context.get("val_score")
        if train_score is not None and val_score is not None:
            gap = train_score - val_score
            if gap > 0.15:
                warnings.append(
                    f"Possible overfitting: training score ({train_score:.3f}) is "
                    f"much higher than validation score ({val_score:.3f}), a gap of "
                    f"{gap:.3f}. Consider regularization, more data, or a simpler model."
                )

        return warnings


if __name__ == "__main__":
    agent = EvaluationAgent()

    rec = agent.recommend_metrics("classification", imbalanced=True,
                                   positive_rate=0.05, fn_costly=True)
    print(rec)

    mistakes = agent.check_common_mistakes(
        "classification",
        used_only_accuracy=True,
        class_balance=0.95,
        train_score=0.99,
        val_score=0.65,
        metrics_reported=["accuracy"],
    )
    for m in mistakes:
        print("-", m)
