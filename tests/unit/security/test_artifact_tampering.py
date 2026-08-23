"""
Artifact tampering tests — verify that the trust system (SHA-256 + Ed25519) correctly
detects:
  1. A valid, untampered bundle → PASS
  2. model.pkl modified after signing → FAIL (integrity)
  3. metrics.json modified after signing → FAIL (integrity)
  4. signature.sig replaced with a random value → FAIL (authenticity)

These tests are part of the CI quality gate and must all pass.
"""
import json
import os
import shutil
import tempfile
import pytest
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
import numpy as np


@pytest.fixture
def trained_model_and_data():
    """Return a tiny fitted LogisticRegression and sample data."""
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    import pandas as pd
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    model = LogisticRegression(random_state=42, max_iter=200)
    model.fit(X_df, y)
    return model, list(X_df.columns)


@pytest.fixture
def artifact_bundle(trained_model_and_data, tmp_path, monkeypatch):
    """
    Create a signed artifact bundle in a temp directory.
    Monkeypatches ARTIFACTS_DIR to tmp_path so tests don't write to the real artifacts/.
    """
    # Patch the ARTIFACTS_DIR constant so bundle is created in tmp_path
    import src.agentic_ml.security.manifest as manifest_mod
    import src.agentic_ml.security.crypto as crypto_mod

    original_root = manifest_mod.ARTIFACTS_ROOT
    manifest_mod.ARTIFACTS_ROOT = tmp_path
    crypto_mod.KEYS_DIR = tmp_path / "keys"

    model, feature_columns = trained_model_and_data

    from src.agentic_ml.security.manifest import ArtifactBundleManager
    bundle_info = ArtifactBundleManager.create_bundle(
        model_name="test_model",
        model_obj=model,
        task_type="classification",
        feature_columns=feature_columns,
        target_column="target",
        metrics={"LogisticRegression": 0.88},
        provenance=[],
        description="Test bundle for tampering tests",
        run_id="run_test_001",
        dataset_hash="abc123",
        random_seed=42,
    )

    yield bundle_info

    manifest_mod.ARTIFACTS_ROOT = original_root


class TestArtifactIntegrity:
    """Tests that verify bundle integrity detection."""

    def test_valid_bundle_passes_verification(self, artifact_bundle):
        """An untampered bundle must pass both integrity and authenticity checks."""
        from src.agentic_ml.security.manifest import ArtifactBundleManager
        result = ArtifactBundleManager.verify_bundle(artifact_bundle["bundle_dir"])
        assert result["valid"] is True, f"Expected valid=True, got errors: {result['errors']}"
        assert result["integrity_ok"] is True
        assert result["signature_ok"] is True
        assert result["errors"] == []

    def test_modified_model_pkl_fails_integrity(self, artifact_bundle):
        """Modifying model.pkl must fail the SHA-256 integrity check."""
        bundle_dir = Path(artifact_bundle["bundle_dir"])
        model_file = bundle_dir / "model.pkl"

        # Tamper: append garbage bytes to model.pkl
        with open(model_file, "ab") as f:
            f.write(b"\x00\xff\x00TAMPERED")

        from src.agentic_ml.security.manifest import ArtifactBundleManager
        result = ArtifactBundleManager.verify_bundle(str(bundle_dir))

        assert result["valid"] is False, "Tampered model.pkl should fail verification"
        assert result["integrity_ok"] is False
        assert any("model.pkl" in e for e in result["errors"]), (
            f"Expected 'model.pkl' in errors, got: {result['errors']}"
        )

    def test_modified_metrics_json_fails_integrity(self, artifact_bundle):
        """Modifying metrics.json must fail the SHA-256 integrity check."""
        bundle_dir = Path(artifact_bundle["bundle_dir"])
        metrics_file = bundle_dir / "metrics.json"

        # Tamper: overwrite with inflated metrics
        metrics_file.write_text(
            json.dumps({"LogisticRegression": 0.999, "TAMPERED": True}, indent=2),
            encoding="utf-8",
        )

        from src.agentic_ml.security.manifest import ArtifactBundleManager
        result = ArtifactBundleManager.verify_bundle(str(bundle_dir))

        assert result["valid"] is False, "Tampered metrics.json should fail verification"
        assert result["integrity_ok"] is False

    def test_replaced_signature_fails_authenticity(self, artifact_bundle):
        """Replacing signature.sig with a random base64 value must fail authenticity."""
        import base64
        bundle_dir = Path(artifact_bundle["bundle_dir"])
        sig_file = bundle_dir / "signature.sig"

        # Tamper: replace with random 64-byte signature
        random_sig = base64.b64encode(os.urandom(64)).decode("utf-8")
        sig_file.write_text(random_sig, encoding="utf-8")

        from src.agentic_ml.security.manifest import ArtifactBundleManager
        result = ArtifactBundleManager.verify_bundle(str(bundle_dir))

        assert result["valid"] is False, "Forged signature should fail authenticity"
        assert result["signature_ok"] is False

    def test_missing_manifest_fails(self, artifact_bundle):
        """A bundle with a missing manifest.json must fail immediately."""
        bundle_dir = Path(artifact_bundle["bundle_dir"])
        (bundle_dir / "manifest.json").unlink()

        from src.agentic_ml.security.manifest import ArtifactBundleManager
        result = ArtifactBundleManager.verify_bundle(str(bundle_dir))

        assert result["valid"] is False
        assert "manifest.json missing" in result["errors"]

    def test_modified_manifest_fails_authenticity(self, artifact_bundle):
        """Modifying manifest.json directly must fail the Ed25519 signature verification."""
        bundle_dir = Path(artifact_bundle["bundle_dir"])
        manifest_file = bundle_dir / "manifest.json"

        manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
        manifest_data["model_name"] = "tampered_model_name"
        manifest_file.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

        from src.agentic_ml.security.manifest import ArtifactBundleManager
        result = ArtifactBundleManager.verify_bundle(str(bundle_dir))
        assert result["valid"] is False
        assert result["signature_ok"] is False

    def test_incomplete_provenance_fails_creation(self, trained_model_and_data, tmp_path):
        """Bundle creation must fail if mandatory provenance is missing or 'unknown'."""
        model, feature_columns = trained_model_and_data
        from src.agentic_ml.security.manifest import ArtifactBundleManager, validate_provenance

        # Missing run_id raises ValueError
        with pytest.raises(ValueError):
            ArtifactBundleManager.create_bundle(
                model_name="test_model",
                model_obj=model,
                task_type="classification",
                run_id="",
                dataset_hash="abc123sha",
            )

        # 'unknown' dataset_hash raises ValueError
        with pytest.raises(ValueError):
            ArtifactBundleManager.create_bundle(
                model_name="test_model",
                model_obj=model,
                task_type="classification",
                run_id="run_123",
                dataset_hash="unknown",
            )

