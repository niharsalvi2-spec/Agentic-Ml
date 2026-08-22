"""
Unit tests for categorical encoding engine.
"""

import unittest
import pandas as pd
import numpy as np
from src.agentic_ml.ml_engine.preprocessing.encoder import (
    classify_cardinality,
    fit_kfold_target_encoder,
    apply_target_encoder,
    fit_frequency_encoder,
    apply_frequency_encoder,
    align_test_columns,
    FeatureEncoder
)


class TestEncoder(unittest.TestCase):

    def test_classify_cardinality(self):
        s1 = pd.Series(["A", "B", "C"])
        self.assertEqual(classify_cardinality(s1), "low")

        s2 = pd.Series([f"cat_{i}" for i in range(30)])
        self.assertEqual(classify_cardinality(s2), "medium")

    def test_kfold_target_encoder(self):
        df = pd.DataFrame({
            "city": ["NY", "NY", "LA", "LA", "CHI", "CHI"] * 10,
            "target": [1, 1, 0, 0, 1, 0] * 10
        })
        train_enc, fmap, gmean = fit_kfold_target_encoder(df, "city", "target", n_splits=3)
        self.assertEqual(len(train_enc), len(df))
        self.assertFalse(train_enc.isnull().any())

        test_df = pd.DataFrame({"city": ["NY", "LA", "Unknown"]})
        test_enc = apply_target_encoder(test_df, "city", fmap, gmean)
        self.assertEqual(len(test_enc), 3)
        self.assertEqual(test_enc.iloc[2], gmean)

    def test_frequency_encoder(self):
        df = pd.DataFrame({"category": ["A", "A", "A", "B"]})
        fmap = fit_frequency_encoder(df, "category")
        encoded = apply_frequency_encoder(df, "category", fmap)
        self.assertEqual(encoded.iloc[0], 0.75)
        self.assertEqual(encoded.iloc[-1], 0.25)

    def test_feature_encoder_class(self):
        df = pd.DataFrame({
            "cat": ["A", "B", "A"],
            "num": [1, 2, 3]
        })
        encoder = FeatureEncoder(method="onehot")
        df_enc = encoder.fit_transform(df)
        self.assertIn("cat_A", df_enc.columns)
        self.assertIn("cat_B", df_enc.columns)


if __name__ == "__main__":
    unittest.main()
