"""
Unit tests for model registry, recommendations, training, and benchmarking.
"""

import unittest
import pandas as pd
import numpy as np
from src.agentic_ml.ml_engine.models.registry import ModelRegistry
from src.agentic_ml.ml_engine.models.training import ModelTrainer
from src.agentic_ml.ml_engine.models.tuning import HyperparameterTuner
from src.agentic_ml.ml_engine.data.loader import DataLoader


class TestModels(unittest.TestCase):

    def test_model_registry_recommendations(self):
        recs = ModelRegistry.recommend(
            task_type="classification",
            n_samples=500,
            n_features=10,
            need_interpretability=True,
            suspect_nonlinear=True
        )
        self.assertGreater(len(recs), 0)
        self.assertIn("model", recs[0])

    def test_model_trainer_train_candidates(self):
        df, target = DataLoader.load_or_synthesize("classification", n_samples=60)
        X = df.drop(columns=[target])
        y = df[target]

        trained = ModelTrainer.train_candidates(X, y, task_type="classification")
        self.assertIn("RandomForest", trained)
        self.assertIn("LogisticRegression", trained)

    def test_model_trainer_compare_all(self):
        df, target = DataLoader.load_or_synthesize("classification", n_samples=80)
        X_train, y_train = df.iloc[:50].drop(columns=[target]), df.iloc[:50][target]
        X_test, y_test = df.iloc[50:].drop(columns=[target]), df.iloc[50:][target]

        benchmarks = ModelTrainer.compare_all(X_train, y_train, X_test, y_test, task_type="classification")
        self.assertGreater(len(benchmarks), 0)
        self.assertIn("accuracy", benchmarks[0])
        self.assertIn("train_time_sec", benchmarks[0])


if __name__ == "__main__":
    unittest.main()
