"""
Tests for Strongly Typed RunContext, Artifact Verification, and Path Traversal Security (Phase 5, 6, 20, 21).
"""
import pytest
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.agentic_ml.core.context import RunContext
from src.agentic_ml.security.manifest import ArtifactBundleManager
from src.agentic_ml.security.path_sanitizer import sanitize_dataset_path


class TestProvenanceAndSecurity:

    def test_run_context_rejects_missing_mandatory_fields(self):
        # Empty run_id
        with pytest.raises(ValueError):
            RunContext.create(run_id="", dataset_hash="abc123sha")

        # Unknown run_id
        with pytest.raises(ValueError):
            RunContext.create(run_id="unknown", dataset_hash="abc123sha")

        # Empty dataset_hash
        with pytest.raises(ValueError):
            RunContext.create(run_id="run_123", dataset_hash="")

        # Unknown dataset_hash
        with pytest.raises(ValueError):
            RunContext.create(run_id="run_123", dataset_hash="unknown")

    def test_run_context_valid_instantiation(self):
        ctx = RunContext.create(run_id="run_20260823_1010", dataset_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
        assert ctx.run_id == "run_20260823_1010"
        assert len(ctx.dataset_hash) == 64
        assert ctx.python_version != ""

    def test_artifact_bundle_creation_and_strict_verification(self, tmp_path):
        model = LogisticRegression()
        X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
        y = np.array([0, 0, 1, 1])
        model.fit(X, y)

        bundle = ArtifactBundleManager.create_bundle(
            model_name="UnitTestingClassifier",
            model_obj=model,
            task_type="classification",
            feature_columns=["feat_a", "feat_b"],
            target_column="target",
            metrics={"accuracy": 1.0, "f1": 1.0},
            run_id="run_sec_test_001",
            dataset_hash="7f83b1657ff1fc53b92dc18148a1d65dfc2d4b1fa3d677284addd200126d9069",
        )

        assert bundle["version"].startswith("v")
        assert Path(bundle["manifest_path"]).exists()
        assert Path(bundle["signature_path"]).exists()

        # Immediate verification
        verification = ArtifactBundleManager.verify_bundle(bundle["bundle_dir"])
        assert verification["valid"] is True
        assert verification["integrity_ok"] is True
        assert verification["signature_ok"] is True
        assert verification["model_load_ok"] is True

    def test_path_sanitizer_blocks_directory_traversal(self):
        # Path traversal with ..
        with pytest.raises(ValueError) as exc:
            sanitize_dataset_path("../../etc/passwd")
        assert "Path traversal detected" in str(exc.value)

        # Path traversal escaping allowed boundaries
        with pytest.raises(ValueError):
            sanitize_dataset_path("C:\\Windows\\System32\\cmd.exe")

        # Null byte injection
        with pytest.raises(ValueError):
            sanitize_dataset_path("data/clean.csv\0.exe")
