"""
Tests for Data Leakage Prevention, Detection, and Transformation Invariants (Phase 7).
"""
import numpy as np
import pandas as pd
import pytest

from src.agentic_ml.ml_engine.data.leakage_detector import LeakageDetector, LeakageReport
from src.agentic_ml.ml_engine.evaluation.validation import EvaluationAgent
from src.agentic_ml.ml_engine.preprocessing.encoder import fit_kfold_target_encoder, apply_target_encoder


class TestMLLeakageAudit:

    def test_leakage_detector_catches_preprocessor_fit_before_split(self):
        X_train = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [4.0, 5.0, 6.0]})
        X_test = pd.DataFrame({"f1": [7.0, 8.0], "f2": [9.0, 10.0]})
        y_train = pd.Series([0, 1, 0])

        report = LeakageDetector.check(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            preprocessor_fit_before_split=True,
        )
        assert report.passed is False
        assert report.severity == "HIGH"
        assert any("CRITICAL: Preprocessor was fitted on the full dataset" in f for f in report.findings)

    def test_leakage_detector_catches_row_overlap(self):
        # Row [1.0, 4.0] is present in both X_train and X_test
        X_train = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [4.0, 5.0, 6.0]})
        X_test = pd.DataFrame({"f1": [1.0, 8.0], "f2": [4.0, 10.0]})
        y_train = pd.Series([0, 1, 0])

        report = LeakageDetector.check(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            preprocessor_fit_before_split=False,
        )
        assert any("appear in both train and test sets" in f for f in report.findings)

    def test_leakage_detector_catches_target_correlated_leak(self):
        # Feature perfectly correlated with target
        y_train = pd.Series([0, 1, 0, 1, 0, 1])
        X_train = pd.DataFrame({
            "normal_feat": [1.2, 3.4, 2.1, 5.5, 1.8, 4.9],
            "leaked_target_feat": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],  # Correlation = 1.0
        })
        X_test = pd.DataFrame({
            "normal_feat": [2.0, 4.0],
            "leaked_target_feat": [0.0, 1.0],
        })

        report = LeakageDetector.check(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            target_column="target",
        )
        assert any("leaked_target_feat" in f and "HIGH RISK" in f for f in report.findings)

    def test_evaluation_agent_flags_scaler_leakage(self):
        eval_agent = EvaluationAgent()
        warnings = eval_agent.check_common_mistakes(
            task="classification",
            scaler_fit_on="train+test",
            evaluated_on_training_data=False,
        )
        assert any("Data leakage detected. Scalers/encoders were fit on full dataset" in w for w in warnings)

    def test_target_encoder_out_of_fold_isolation(self):
        # Target encoding must be computed out-of-fold for train and applied strictly as map to test
        train_df = pd.DataFrame({
            "category": ["A", "A", "B", "B", "A", "B", "A", "B", "A", "B"],
            "target": [1, 1, 0, 0, 1, 0, 1, 0, 1, 0],
        })
        test_df = pd.DataFrame({
            "category": ["A", "B", "C"],
        })

        encoded_train, final_map, global_mean = fit_kfold_target_encoder(
            train_df=train_df,
            cat_col="category",
            target_col="target",
            n_splits=2,
            smoothing=1.0,
            seed=42,
        )
        assert len(encoded_train) == len(train_df)
        assert not encoded_train.isna().any()

        encoded_test = apply_target_encoder(test_df, "category", final_map, global_mean)
        assert len(encoded_test) == 3
        # Unseen category "C" gets global mean fallback, avoiding NaN or training contamination
        assert encoded_test.iloc[2] == pytest.approx(global_mean)
