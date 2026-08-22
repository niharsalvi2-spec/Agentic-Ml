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


if __name__ == "__main__":
    unittest.main()
