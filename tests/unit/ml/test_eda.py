"""
Unit tests for EDA statistics, distribution analysis, correlations, and outliers.
"""

import unittest
import pandas as pd
import numpy as np
from src.agentic_ml.ml_engine.eda.statistics import recommend_bins, skew_kurtosis, skew_label, EDAEngine
from src.agentic_ml.ml_engine.eda.distributions import DistributionAnalyzer
from src.agentic_ml.ml_engine.eda.correlations import CorrelationAnalyzer
from src.agentic_ml.ml_engine.eda.outliers import OutlierDetector


class TestEDA(unittest.TestCase):

    def test_recommend_bins_freedman_diaconis(self):
        s = pd.Series(np.random.randn(100))
        bins = recommend_bins(s)
        self.assertGreater(bins, 1)

    def test_skew_kurtosis_and_label(self):
        s = pd.Series(np.exp(np.random.randn(100)))  # Log-normal is right-skewed
        sk = skew_kurtosis(s)
        self.assertIn("skewness", sk)
        self.assertIn("kurtosis_excess", sk)
        lbl = skew_label(sk["skewness"])
        self.assertIsInstance(lbl, str)

    def test_eda_engine_analyze(self):
        df = pd.DataFrame({
            "f1": np.random.randn(50),
            "f2": np.random.randn(50),
            "cat": ["A", "B"] * 25
        })
        res = EDAEngine.analyze(df)
        self.assertEqual(res["n_rows"], 50)
        self.assertIn("summary_stats", res)
        self.assertIn("f1", res["skewness_and_kurtosis"])

    def test_correlation_analyzer(self):
        x = np.linspace(0, 10, 50)
        df = pd.DataFrame({
            "x": x,
            "y": x * 2 + np.random.randn(50) * 0.01,
            "target": x * 3
        })
        res = CorrelationAnalyzer.analyze_correlations(df, target_col="target", threshold=0.90)
        self.assertGreaterEqual(len(res["multicollinear_pairs"]), 1)
        self.assertIn("suggest_drop", res["multicollinear_pairs"][0])

    def test_outlier_detector(self):
        s = pd.Series([1, 2, 2, 3, 2, 1, 2, 100])
        res = OutlierDetector.detect_iqr_outliers(s)
        self.assertEqual(res["outlier_count"], 1)


if __name__ == "__main__":
    unittest.main()
