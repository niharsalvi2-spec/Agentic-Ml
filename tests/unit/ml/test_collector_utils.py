"""
Unit tests for data collector utilities and DataLoader.
"""

import unittest
import pandas as pd
import numpy as np
from src.agentic_ml.ml_engine.data.collector_utils import with_retry, RateLimiter, quick_quality_check
from src.agentic_ml.ml_engine.data.loader import DataLoader


class TestCollectorUtils(unittest.TestCase):

    def test_with_retry_success(self):
        attempts = 0

        @with_retry(max_attempts=3, base_delay=0.01)
        def flaky_func():
            nonlocal attempts
            attempts += 1
            if attempts < 2:
                raise ValueError("Transient error")
            return "success"

        result = flaky_func()
        self.assertEqual(result, "success")
        self.assertEqual(attempts, 2)

    def test_rate_limiter(self):
        limiter = RateLimiter(calls_per_second=50)
        limiter.wait()
        # Ensure no exception thrown

    def test_quick_quality_check(self):
        df = pd.DataFrame({
            "a": [1, 2, None, 4],
            "b": ["x", "y", "z", "x"]
        })
        report = quick_quality_check(df)
        self.assertEqual(report["n_rows"], 4)
        self.assertEqual(report["n_columns"], 2)
        self.assertIn("a", report["null_counts"])

    def test_data_loader_formats(self):
        df_cls, target_cls = DataLoader.load_or_synthesize("classification", n_samples=100)
        self.assertEqual(len(df_cls), 100)
        self.assertEqual(target_cls, "target")

        df_reg, target_reg = DataLoader.load_or_synthesize("regression", n_samples=50)
        self.assertEqual(len(df_reg), 50)
        self.assertEqual(target_reg, "target")

        df_cl, target_cl = DataLoader.load_or_synthesize("clustering", n_samples=60)
        self.assertEqual(len(df_cl), 60)


if __name__ == "__main__":
    unittest.main()
