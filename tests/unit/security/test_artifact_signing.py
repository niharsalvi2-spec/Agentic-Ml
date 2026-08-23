"""
Unit tests for Asymmetric Cryptographic Signing & Artifact Bundle Verification.
Verifies ECDSA signing, SHA-256 manifest integrity checks, and tamper detection.
"""
import os
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

from src.agentic_ml.security.crypto import (
    generate_keypair,
    sign_bytes,
    verify_signature,
    get_or_create_signing_keys,
)
from src.agentic_ml.security.manifest import (
    ArtifactBundleManager,
    compute_sha256,
)


class TestArtifactSecurity(unittest.TestCase):

    def setUp(self):
        self.priv_pem, self.pub_pem = generate_keypair()
        self.test_dir = tempfile.mkdtemp(prefix="agentic_ml_sec_test_")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_keypair_generation(self):
        self.assertTrue(self.priv_pem.startswith(b"-----BEGIN PRIVATE KEY-----"))
        self.assertTrue(self.pub_pem.startswith(b"-----BEGIN PUBLIC KEY-----"))

    def test_sign_and_verify_signature(self):
        data = b"Arbitrary binary ML artifact data or manifest"
        sig_b64 = sign_bytes(data, self.priv_pem)
        self.assertIsInstance(sig_b64, str)
        self.assertTrue(verify_signature(data, sig_b64, self.pub_pem))

    def test_corrupted_signature_fails(self):
        data = b"Important artifact data"
        sig_b64 = sign_bytes(data, self.priv_pem)
        # Corrupt signature string
        corrupted_sig = ("A" if sig_b64[0] != "A" else "B") + sig_b64[1:]
        self.assertFalse(verify_signature(data, corrupted_sig, self.pub_pem))

    def test_modified_data_signature_fails(self):
        data = b"Original manifest content"
        sig_b64 = sign_bytes(data, self.priv_pem)
        modified_data = b"Tampered manifest content"
        self.assertFalse(verify_signature(modified_data, sig_b64, self.pub_pem))

    def test_untrusted_public_key_fails(self):
        data = b"Sensitive model manifest"
        sig_b64 = sign_bytes(data, self.priv_pem)

        # Generate a separate, untrusted keypair
        _, other_pub_pem = generate_keypair()
        self.assertFalse(verify_signature(data, sig_b64, other_pub_pem))

    def test_artifact_bundle_creation_and_verification(self):
        rf = RandomForestClassifier(n_estimators=5, random_state=42)
        rf.fit([[1, 2], [3, 4], [5, 6]], [0, 1, 0])

        bundle_info = ArtifactBundleManager.create_bundle(
            model_name="TestClassifier",
            model_obj=rf,
            task_type="classification",
            feature_columns=["f1", "f2"],
            target_column="target",
            metrics={"accuracy": 0.98, "f1_score": 0.97},
            provenance=[{"agent": "test", "operation": "unit_test"}],
            private_key_pem=self.priv_pem,
            run_id="run_test_001",
            dataset_hash="7f83b1657ff1fc53b92dc18148a1d65dfc2d4b1fa3d677284addd200126d9069",
        )

        bundle_dir = bundle_info["bundle_dir"]
        self.assertTrue(os.path.exists(os.path.join(bundle_dir, "model.pkl")))
        self.assertTrue(os.path.exists(os.path.join(bundle_dir, "manifest.json")))
        self.assertTrue(os.path.exists(os.path.join(bundle_dir, "signature.sig")))
        self.assertTrue(os.path.exists(os.path.join(bundle_dir, "schema.json")))
        self.assertTrue(os.path.exists(os.path.join(bundle_dir, "metrics.json")))
        self.assertTrue(os.path.exists(os.path.join(bundle_dir, "provenance.json")))

        # Verification using matching public key
        ver_result = ArtifactBundleManager.verify_bundle(bundle_dir, public_key_pem=self.pub_pem)
        self.assertTrue(ver_result["valid"])
        self.assertTrue(ver_result["integrity_ok"])
        self.assertTrue(ver_result["signature_ok"])
        self.assertEqual(len(ver_result["errors"]), 0)

    def test_bundle_tamper_detection_on_model(self):
        rf = RandomForestClassifier(n_estimators=5, random_state=42)
        rf.fit([[1, 2], [3, 4]], [0, 1])

        bundle_info = ArtifactBundleManager.create_bundle(
            model_name="TamperTestModel",
            model_obj=rf,
            task_type="classification",
            private_key_pem=self.priv_pem,
            run_id="run_test_002",
            dataset_hash="7f83b1657ff1fc53b92dc18148a1d65dfc2d4b1fa3d677284addd200126d9069",
        )
        bundle_dir = bundle_info["bundle_dir"]

        # Tamper with model.pkl on disk
        model_file = Path(bundle_dir) / "model.pkl"
        with open(model_file, "ab") as f:
            f.write(b"\x00TAMPERED_PAYLOAD\x00")

        ver_result = ArtifactBundleManager.verify_bundle(bundle_dir, public_key_pem=self.pub_pem)
        self.assertFalse(ver_result["valid"])
        self.assertFalse(ver_result["integrity_ok"])
        self.assertTrue(any("Hash mismatch in model.pkl" in err for err in ver_result["errors"]))

    def test_bundle_tamper_detection_on_manifest(self):
        rf = RandomForestClassifier(n_estimators=5, random_state=42)
        rf.fit([[1, 2], [3, 4]], [0, 1])

        bundle_info = ArtifactBundleManager.create_bundle(
            model_name="ManifestTamperModel",
            model_obj=rf,
            task_type="classification",
            private_key_pem=self.priv_pem,
            run_id="run_test_003",
            dataset_hash="7f83b1657ff1fc53b92dc18148a1d65dfc2d4b1fa3d677284addd200126d9069",
        )
        bundle_dir = bundle_info["bundle_dir"]

        # Tamper with manifest.json without updating signature
        manifest_file = Path(bundle_dir) / "manifest.json"
        data = json.loads(manifest_file.read_text(encoding="utf-8"))
        data["model_name"] = "ForgedModelName"
        manifest_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

        ver_result = ArtifactBundleManager.verify_bundle(bundle_dir, public_key_pem=self.pub_pem)
        self.assertFalse(ver_result["valid"])
        self.assertFalse(ver_result["signature_ok"])
        self.assertTrue(any("Digital signature invalid" in err for err in ver_result["errors"]))


if __name__ == "__main__":
    unittest.main()
