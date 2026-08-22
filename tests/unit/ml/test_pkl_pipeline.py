"""
Unit tests for PKL serialization, SHA-256 integrity hashing, PKLBundleLoader, and PKLVersionManager.
"""

import unittest
import tempfile
import os
import shutil
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.agentic_ml.ml_engine.pipelines.pkl_utils import (
    save_pkl,
    load_pkl,
    compute_file_hash,
    save_pkl_with_hash,
    safe_load_pkl,
    PKLSecurityError,
    PKLVersionManager
)
from src.agentic_ml.ml_engine.pipelines.artifact_pipeline import PKLGeneratorAgent, PKLBundleLoader


class TestPKLPipeline(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.X = pd.DataFrame(np.random.randn(50, 3), columns=["f1", "f2", "f3"])
        self.y = pd.Series((self.X["f1"] + self.X["f2"] > 0).astype(int))
        self.model = RandomForestClassifier(n_estimators=10, random_state=42).fit(self.X, self.y)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_save_and_safe_load_with_hash(self):
        pkl_path = os.path.join(self.tmp_dir, "test_model.pkl")
        path, file_hash = save_pkl_with_hash(self.model, pkl_path)
        self.assertTrue(os.path.exists(path))
        self.assertTrue(os.path.exists(path + ".hash"))

        loaded = safe_load_pkl(path, verify_hash=True)
        self.assertEqual(len(loaded.predict(self.X)), 50)

    def test_tamper_detection_raises_security_error(self):
        pkl_path = os.path.join(self.tmp_dir, "tamper_test.pkl")
        path, _ = save_pkl_with_hash(self.model, pkl_path)

        # Corrupt file content
        with open(path, "ab") as f:
            f.write(b"corrupted_bytes")

        with self.assertRaises(PKLSecurityError):
            safe_load_pkl(path, verify_hash=True)

    def test_pkl_generator_and_bundle_loader(self):
        generator = PKLGeneratorAgent(save_dir=self.tmp_dir)
        gen_res = generator.generate(
            pipeline_or_model=self.model,
            task="classification",
            model_name="loan_classifier",
            feature_columns=["f1", "f2", "f3"],
            metrics={"accuracy": 0.95},
            register_version=True
        )

        self.assertTrue(os.path.exists(gen_res["filepath"]))
        loader = PKLGeneratorAgent.load(gen_res["filepath"])
        preds = loader.predict(self.X)
        self.assertEqual(len(preds), 50)

        summary = loader.summary()
        self.assertEqual(summary["model_name"], "loan_classifier")
        self.assertEqual(summary["metrics"]["accuracy"], 0.95)


if __name__ == "__main__":
    unittest.main()
