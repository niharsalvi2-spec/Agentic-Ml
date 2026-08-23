"""
Unit tests for zero-dependency metrics and EvaluationAgent.
"""

import unittest
import numpy as np
from src.agentic_ml.ml_engine.evaluation.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    regression_report,
    silhouette_score,
    davies_bouldin_index,
    clustering_report
)
from src.agentic_ml.ml_engine.evaluation.validation import EvaluationAgent, ModelEvaluator


class TestMetricsAndEvaluation(unittest.TestCase):

    def test_classification_metrics(self):
        y_true = [1, 1, 0, 0, 1, 0]
        y_pred = [1, 0, 0, 0, 1, 1]

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        self.assertAlmostEqual(acc, 4 / 6)
        self.assertGreater(prec, 0.0)
        self.assertGreater(rec, 0.0)
        self.assertGreater(f1, 0.0)

        rep = classification_report(y_true, y_pred)
        self.assertIn("confusion_matrix", rep)
        self.assertIn("accuracy", rep)

    def test_regression_metrics(self):
        y_true = [10.0, 20.0, 30.0]
        y_pred = [12.0, 19.0, 28.0]

        mae = mean_absolute_error(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        self.assertAlmostEqual(mae, (2.0 + 1.0 + 2.0) / 3.0)
        self.assertGreater(r2, 0.5)

        rep = regression_report(y_true, y_pred, n_features=1)
        self.assertIn("adjusted_r2", rep)

    def test_clustering_metrics(self):
        X = np.array([[0, 0], [0.1, 0.1], [5, 5], [5.1, 5.1]])
        labels = [0, 0, 1, 1]

        sil = silhouette_score(X, labels)
        db = davies_bouldin_index(X, labels)

        self.assertGreater(sil, 0.8)
        self.assertLess(db, 0.5)

        rep = clustering_report(X, labels)
        self.assertIn("silhouette", rep)

    def test_evaluation_agent_mistake_checks(self):
        agent = EvaluationAgent()
        warnings = agent.check_common_mistakes(
            task="classification",
            evaluated_on_training_data=True,
            scaler_fit_on="train+test",
            class_balance=0.95,
            used_only_accuracy=True
        )
        self.assertGreaterEqual(len(warnings), 2)

    def test_extract_primary_metric_classification_and_regression(self):
        from src.agentic_ml.ml_engine.evaluation.metrics import extract_primary_metric, MetricRegistry

        # Classification
        clf_metrics = {"accuracy": 0.88, "f1": 0.85, "precision": 0.86}
        pm_clf = extract_primary_metric(clf_metrics, task_type="classification")
        self.assertEqual(pm_clf.name, "f1")
        self.assertEqual(pm_clf.direction, "maximize")
        self.assertAlmostEqual(pm_clf.value, 0.85)

        # Regression
        reg_metrics = {"rmse": 0.45, "mae": 0.32, "r2": 0.89}
        pm_reg = extract_primary_metric(reg_metrics, task_type="regression")
        self.assertEqual(pm_reg.name, "r2")
        self.assertEqual(pm_reg.direction, "maximize")

        # Regression error-only metrics
        reg_err_metrics = {"rmse": 0.45, "mae": 0.32}
        pm_reg_err = extract_primary_metric(reg_err_metrics, task_type="regression")
        self.assertEqual(pm_reg_err.name, "rmse")
        self.assertEqual(pm_reg_err.direction, "minimize")

        # Empty metrics raises ValueError
        with self.assertRaises(ValueError):
            extract_primary_metric({}, task_type="classification")

        # Registry checks
        self.assertTrue(MetricRegistry.validate_metric_for_task("f1", "classification"))
        self.assertFalse(MetricRegistry.validate_metric_for_task("rmse", "classification"))
        self.assertTrue(MetricRegistry.validate_metric_for_task("rmse", "regression"))

    def test_single_class_edge_case_classification_report(self):
        y_true = [1, 1, 1, 1]
        y_pred = [1, 1, 1, 1]
        rep = classification_report(y_true, y_pred)
        self.assertEqual(rep["accuracy"], 1.0)
        self.assertEqual(rep["precision"], 1.0)
        self.assertEqual(rep["recall"], 1.0)
        self.assertEqual(rep["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()

