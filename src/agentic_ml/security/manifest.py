"""
ArtifactBundleManifest & Manifest Manager.
Manages manifest generation, SHA-256 hashing across all bundle components,
asymmetric signing, and independent verification.
"""
import os
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple, Sequence

import joblib

from src.agentic_ml.core.constants import ARTIFACTS_DIR
from src.agentic_ml.security.crypto import (
    sign_bytes,
    verify_signature,
    get_or_create_signing_keys,
)

logger = logging.getLogger("agentic_ml.security.manifest")

ARTIFACTS_ROOT = ARTIFACTS_DIR



def compute_sha256(filepath: str) -> str:
    """Compute SHA-256 hexadecimal digest for a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class ArtifactBundleManager:
    """
    Manages generation, signing, and verification of immutable ML artifact bundles.

    Bundle layout:
      artifacts/<model_name>/v<N>/
        ├── model.pkl
        ├── schema.json
        ├── metrics.json
        ├── provenance.json
        ├── manifest.json
        └── signature.sig
    """

    @staticmethod
    def _get_next_version_dir(model_dir: Path) -> Path:
        """Find next available version directory (v1, v2, ...)."""
        model_dir.mkdir(parents=True, exist_ok=True)
        version = 1
        while (model_dir / f"v{version}").exists():
            version += 1
        target = model_dir / f"v{version}"
        target.mkdir(parents=True, exist_ok=True)
        return target

    @classmethod
    def create_bundle(
        cls,
        model_name: str,
        model_obj: Any,
        task_type: str,
        feature_columns: Optional[List[str]] = None,
        target_column: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        provenance: Optional[Sequence[Any]] = None,
        private_key_pem: Optional[bytes] = None,
        description: str = "",
    ) -> Dict[str, Any]:
        """
        Create, hash, and digitally sign a complete ML artifact bundle.
        """
        safe_name = model_name.lower().replace(" ", "_")
        model_dir = ARTIFACTS_ROOT / safe_name
        version_dir = cls._get_next_version_dir(model_dir)

        # 1. Save model.pkl
        model_path = version_dir / "model.pkl"
        joblib.dump(model_obj, str(model_path), compress=3)

        # 2. Save schema.json
        schema_data = {
            "model_name": model_name,
            "task_type": task_type,
            "feature_columns": feature_columns or [],
            "target_column": target_column or "",
            "description": description,
        }
        schema_path = version_dir / "schema.json"
        schema_path.write_text(json.dumps(schema_data, indent=2), encoding="utf-8")

        # 3. Save metrics.json
        metrics_data = metrics or {}
        metrics_path = version_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics_data, indent=2), encoding="utf-8")

        # 4. Save provenance.json
        provenance_data = provenance or []
        provenance_path = version_dir / "provenance.json"
        provenance_path.write_text(json.dumps(provenance_data, indent=2), encoding="utf-8")

        # 5. Compute SHA-256 for all constituent files
        file_hashes = {
            "model.pkl": compute_sha256(str(model_path)),
            "schema.json": compute_sha256(str(schema_path)),
            "metrics.json": compute_sha256(str(metrics_path)),
            "provenance.json": compute_sha256(str(provenance_path)),
        }

        # 6. Create manifest.json
        manifest_data = {
            "model_name": model_name,
            "task_type": task_type,
            "version": version_dir.name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "description": description,
            "files": file_hashes,
        }
        manifest_bytes = json.dumps(manifest_data, indent=2, sort_keys=True).encode("utf-8")
        manifest_path = version_dir / "manifest.json"
        manifest_path.write_bytes(manifest_bytes)

        # 7. Asymmetrically sign manifest.json -> signature.sig
        if not private_key_pem:
            priv_pem, _ = get_or_create_signing_keys()
        else:
            priv_pem = private_key_pem

        sig_b64 = sign_bytes(manifest_bytes, priv_pem)
        sig_path = version_dir / "signature.sig"
        sig_path.write_text(sig_b64, encoding="utf-8")

        logger.info(
            "Artifact Bundle created at %s [Version: %s, SHA-256(model): %s...]",
            version_dir, version_dir.name, file_hashes["model.pkl"][:12]
        )

        return {
            "bundle_dir": str(version_dir),
            "model_path": str(model_path),
            "manifest_path": str(manifest_path),
            "signature_path": str(sig_path),
            "version": version_dir.name,
            "hashes": file_hashes,
            "manifest": manifest_data,
        }

    @classmethod
    def verify_bundle(
        cls,
        bundle_dir: str,
        public_key_pem: Optional[bytes] = None,
    ) -> Dict[str, Any]:
        """
        Verify bundle integrity and cryptographic authenticity.

        1. Verifies SHA-256 of all constituent files against manifest.json.
        2. Verifies digital signature in signature.sig against manifest.json using public key.
        """
        b_path = Path(bundle_dir)
        manifest_file = b_path / "manifest.json"
        sig_file = b_path / "signature.sig"

        errors: List[str] = []
        if not manifest_file.exists():
            return {"valid": False, "integrity_ok": False, "signature_ok": False, "errors": ["manifest.json missing"]}
        if not sig_file.exists():
            return {"valid": False, "integrity_ok": False, "signature_ok": False, "errors": ["signature.sig missing"]}

        manifest_bytes = manifest_file.read_bytes()
        try:
            manifest_data = json.loads(manifest_bytes.decode("utf-8"))
        except Exception as exc:
            return {"valid": False, "integrity_ok": False, "signature_ok": False, "errors": [f"Malformed manifest: {exc}"]}

        # 1. Integrity check: verify hashes of all constituent files
        integrity_ok = True
        files_map = manifest_data.get("files", {})
        for filename, expected_hash in files_map.items():
            target_file = b_path / filename
            if not target_file.exists():
                integrity_ok = False
                errors.append(f"Missing bundle component: {filename}")
                continue
            actual_hash = compute_sha256(str(target_file))
            if actual_hash != expected_hash:
                integrity_ok = False
                errors.append(f"Hash mismatch in {filename}: expected {expected_hash}, got {actual_hash}")

        # 2. Authenticity check: verify digital signature
        if not public_key_pem:
            _, pub_pem = get_or_create_signing_keys()
        else:
            pub_pem = public_key_pem

        sig_b64 = sig_file.read_text(encoding="utf-8").strip()
        signature_ok = verify_signature(manifest_bytes, sig_b64, pub_pem)
        if not signature_ok:
            errors.append("Digital signature invalid or untrusted public key")

        valid = integrity_ok and signature_ok

        return {
            "valid": valid,
            "integrity_ok": integrity_ok,
            "signature_ok": signature_ok,
            "errors": errors,
            "manifest": manifest_data,
        }
