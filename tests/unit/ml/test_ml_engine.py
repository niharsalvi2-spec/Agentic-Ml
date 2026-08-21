import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from src.agentic_ml.ml_engine.data.loader import DataLoader
from src.agentic_ml.ml_engine.data.profiler import DataProfiler
from src.agentic_ml.ml_engine.preprocessing.cleaner import DeterministicPreprocessor
from src.agentic_ml.ml_engine.eda.statistics import EDAEngine
from src.agentic_ml.ml_engine.features.selection import FeatureSelector
from src.agentic_ml.ml_engine.models.training import ModelTrainer
from src.agentic_ml.ml_engine.evaluation.validation import ModelEvaluator
from src.agentic_ml.ml_engine.pipelines.artifact_pipeline import ArtifactSerializer

class TestMLEngine(unittest.TestCase):
    def setUp(self):
        self.df, self.target = DataLoader.load_or_synthesize("classification")

    def test_data_loader_and_profiler(self):
        self.assertIsInstance(self.df, pd.DataFrame)
        self.assertGreater(self.df.shape[0], 50)
        profile = DataProfiler.profile(self.df, self.target)
        self.assertEqual(profile["target_column"], self.target)
        self.assertIn("n_rows", profile)

    def test_preprocessing_and_eda(self):
        preprocessor = DeterministicPreprocessor()
        X, y = preprocessor.fit_transform(self.df, self.target)
        self.assertEqual(len(X), len(y))
        self.assertEqual(X.isnull().sum().sum(), 0)
        
        eda_res = EDAEngine.analyze(self.df)
        self.assertIn("summary_stats", eda_res)

    def test_feature_selection_and_training(self):
        preprocessor = DeterministicPreprocessor()
        X, y = preprocessor.fit_transform(self.df, self.target)
        selected = FeatureSelector.select_top_k(X, y, task_type="classification", k=4)
        self.assertEqual(len(selected), 4)

        trained = ModelTrainer.train_candidates(X[selected], y, task_type="classification")
        self.assertIn("RandomForest", trained)
        
        best_name, scores = ModelEvaluator.evaluate(trained, X[selected], y, task_type="classification")
        self.assertIn(best_name, trained)
        self.assertGreater(scores[best_name], 0.5)

    def test_artifact_serialization(self):
        preprocessor = DeterministicPreprocessor()
        X, y = preprocessor.fit_transform(self.df, self.target)
        trained = ModelTrainer.train_candidates(X, y, task_type="classification")
        meta = {"model_name": "RandomForest", "test": True}
        path = ArtifactSerializer.save_artifact(trained["RandomForest"], meta, filename="test_model.pkl")
        self.assertTrue(Path(path).exists())

if __name__ == "__main__":
    unittest.main()
