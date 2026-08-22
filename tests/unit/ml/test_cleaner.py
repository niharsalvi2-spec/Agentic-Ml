"""
Unit tests for preprocessing cleaner, imputer, fences, and outlier detection.
"""

import unittest
import pandas as pd
import numpy as np
from src.agentic_ml.ml_engine.preprocessing.cleaner import (
    DeterministicPreprocessor,
    compute_iqr_fences,
    apply_fences,
    modified_z_outliers,
    dedupe_with_log,
    missingness_report,
    classify_missingness_hint
)
from src.agentic_ml.ml_engine.preprocessing.imputer import DataImputer
from src.agentic_ml.ml_engine.preprocessing.scaler import FeatureScaler


class TestCleanerAndPreprocessor(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            "num1": [1.0, 2.0, np.nan, 4.0, 100.0],
            "num2": [10.0, 20.0, 30.0, 40.0, 50.0],
            "cat": ["A", "B", "A", None, "B"],
            "target": [0, 1, 0, 1, 0]
        })

    def test_missingness_report(self):
        rep = missingness_report(self.df)
        self.assertIn("missing_count", rep.columns)
        self.assertEqual(rep.loc["num1", "missing_count"], 1)

    def test_compute_and_apply_fences(self):
        series = pd.Series([10, 12, 11, 13, 100, 12, 11, 10])
        lower, upper = compute_iqr_fences(series)
        self.assertLess(lower, 10)
        self.assertLess(upper, 100)

        df = pd.DataFrame({"val": series})
        clipped = apply_fences(df, {"val": (lower, upper)})
        self.assertLessEqual(clipped["val"].max(), upper)

    def test_modified_z_outliers(self):
        series = pd.Series([10, 10, 11, 10, 10, 11, 1000])
        mask = modified_z_outliers(series, threshold=3.5)
        self.assertTrue(mask.iloc[-1])

    def test_dedupe_with_log(self):
        df_dupes = pd.DataFrame({
            "k": [1, 1, 2],
            "v": [None, "complete", "x"]
        })
        deduped = dedupe_with_log(df_dupes, subset=["k"])
        self.assertEqual(len(deduped), 2)
        self.assertEqual(deduped.loc[deduped["k"] == 1, "v"].iloc[0], "complete")

    def test_deterministic_preprocessor(self):
        preprocessor = DeterministicPreprocessor(clip_outliers=True)
        X, y = preprocessor.fit_transform(self.df, "target")
        self.assertEqual(len(X), len(y))
        self.assertEqual(X["num1"].isnull().sum(), 0)

    def test_data_imputer_and_scaler(self):
        imputer = DataImputer(numeric_strategy="median")
        df_imputed = imputer.fit_transform(self.df[["num1", "num2"]])
        self.assertEqual(df_imputed.isnull().sum().sum(), 0)

        scaler = FeatureScaler(method="standard")
        scaled = scaler.fit_transform(df_imputed)
        self.assertAlmostEqual(scaled["num2"].mean(), 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
