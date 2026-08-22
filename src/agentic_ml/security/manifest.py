"""
ArtifactBundleManifest & Manifest Manager.

Manages manifest generation, SHA-256 hashing across all bundle components,
Ed25519 signing, and independent verification.

SLSA-aligned provenance design:
  SLSA defines provenance as verifiable information describing how an artifact
  was produced — including its inputs, invocation/build definition, and builder
  context. (https://slsa.dev/spec/v1.2/provenance)

  Our manifest captures:
    - dataset_hash:        sha256 of the raw input dataset
    - run_id:              unique run identifier for reproducibility
    - agent_graph_version: which version of the agent graph produced this
    - python_version:      runtime environment
    - sklearn_version:     key dependency version
    - random_seed:         fixed seed for reproducibility
    - dependency_lock_hash: sha256 of requirements.txt (if available)
    - parent_artifacts:    hashes of all intermediate artifacts in the lineage

Trust model (three distinct concepts):
  - SHA-256        → INTEGRITY   (was the file modified?)
  - Ed25519 sig    → AUTHENTICITY (who signed it? trusted signer?)
  - Provenance JSON → TRACEABILITY (how was it produced?)

Bundle layout:
  artifacts/<model_name>/v<N>/
    ├── model.pkl
    ├── schema.json
    ├── metrics.json
    ├── provenance.json
    ├── manifest.json
    ├── signature.sig
    └── README.json
"""
import os
import sys
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Sequence

import joblib

from src.agentic_ml.core.constants import ARTIFACTS_DIR
from src.agentic_ml.security.crypto import (
    sign_bytes,
    verify_signature,
    get_or_create_signing_keys,
)

logger = logging.getLogger("agentic_ml.security.manifest")

ARTIFACTS_ROOT = ARTIFACTS_DIR
_GRAPH_VERSION = "2.0.0"   # Increment when graph topology changes


def compute_sha256(filepath: str) -> str:
    """Compute SHA-256 hexadecimal digest for a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def _requirements_hash() -> str:
    """Compute SHA-256 of requirements.txt for dependency lock tracking."""
    req_path = Path("requirements.txt")
    if req_path.exists():
        return hashlib.sha256(req_path.read_bytes()).hexdigest()
    return "requirements.txt_not_found"


def _sklearn_version() -> str:
    try:
        import sklearn
        return sklearn.__version__
    except ImportError:
        return "unknown"


class ArtifactBundleManager:
    """
    Manages generation, signing, and verification of immutable ML artifact bundles.

    Integrity  → SHA-256 of every file in the bundle
    Authenticity → Ed25519 signature over manifest.json
    Traceability → provenance.json with full lineage
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
        # SLSA provenance fields
        run_id: Optional[str] = None,
        dataset_hash: Optional[str] = None,
        random_seed: int = 42,
        python_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create, hash, and digitally sign a complete ML artifact bundle.

        Returns dict with:
            bundle_dir, model_path, manifest_path, signature_path,
            version, hashes, manifest
        """
        safe_name = model_name.lower().replace(" ", "_")
        model_dir = ARTIFACTS_ROOT / safe_name
        version_dir = cls._get_next_version_dir(model_dir)

        # 1. Save model.pkl
        model_path = version_dir / "model.pkl"
        joblib.dump(model_obj, str(model_path), compress=3)

        # 2. Attempt ONNX export (best-effort)
        onnx_path = version_dir / "model.onnx"
        onnx_exported = False
        try:
            if feature_columns:
                from skl2onnx import convert_sklearn
                from skl2onnx.common.data_types import FloatTensorType
                initial_type = [("float_input", FloatTensorType([None, len(feature_columns)]))]
                onnx_model = convert_sklearn(model_obj, initial_types=initial_type)
                with open(str(onnx_path), "wb") as f:
                    f.write(onnx_model.SerializeToString())
                onnx_exported = True
                logger.info("ONNX export successful: %s", onnx_path)
        except Exception as exc:
            logger.debug("ONNX export skipped (best-effort): %s", exc)

        # 3. Save schema.json
        schema_data = {
            "model_name": model_name,
            "task_type": task_type,
            "feature_columns": feature_columns or [],
            "target_column": target_column or "",
            "description": description,
            "sklearn_version": _sklearn_version(),
        }
        schema_path = version_dir / "schema.json"
        schema_path.write_text(json.dumps(schema_data, indent=2), encoding="utf-8")

        # 4. Save metrics.json
        metrics_path = version_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics or {}, indent=2), encoding="utf-8")

        # 5. Save provenance.json (pipeline audit trail)
        provenance_path = version_dir / "provenance.json"
        provenance_path.write_text(
            json.dumps(list(provenance or []), indent=2, default=str),
            encoding="utf-8",
        )

        # 6. Compute SHA-256 for all constituent files
        file_hashes: Dict[str, str] = {
            "model.pkl": compute_sha256(str(model_path)),
            "schema.json": compute_sha256(str(schema_path)),
            "metrics.json": compute_sha256(str(metrics_path)),
            "provenance.json": compute_sha256(str(provenance_path)),
        }
        if onnx_exported:
            file_hashes["model.onnx"] = compute_sha256(str(onnx_path))

        # 7. Create SLSA-aligned manifest.json
        manifest_data = {
            # Artifact identity
            "model_name": model_name,
            "task_type": task_type,
            "version": version_dir.name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "description": description,
            # File integrity (SHA-256 per file)
            "files": file_hashes,
            # SLSA provenance fields
            "run_id": run_id or "unknown",
            "dataset_hash": dataset_hash or "unknown",
            "agent_graph_version": _GRAPH_VERSION,
            "python_version": python_version or sys.version,
            "sklearn_version": _sklearn_version(),
            "random_seed": random_seed,
            "dependency_lock_hash": _requirements_hash(),
            # Signing metadata
            "signing_algorithm": "Ed25519",
            "integrity_algorithm": "SHA-256",
        }
        manifest_bytes = json.dumps(manifest_data, indent=2, sort_keys=True).encode("utf-8")
        manifest_path = version_dir / "manifest.json"
        manifest_path.write_bytes(manifest_bytes)

        # 8. Ed25519 sign manifest.json → signature.sig
        if not private_key_pem:
            priv_pem, _ = get_or_create_signing_keys()
        else:
            priv_pem = private_key_pem

        sig_b64 = sign_bytes(manifest_bytes, priv_pem)
        sig_path = version_dir / "signature.sig"
        sig_path.write_text(sig_b64, encoding="utf-8")

        # 9. Write human-readable README.json
        readme = {
            "artifact_id": f"{safe_name}/{version_dir.name}",
            "model_name": model_name,
            "version": version_dir.name,
            "created_at": manifest_data["created_at"],
            "task_type": task_type,
            "integrity_status": "SHA-256 per-file hashes stored in manifest.json",
            "authenticity_status": "Ed25519 signature over manifest.json in signature.sig",
            "provenance_status": "Full pipeline audit trail in provenance.json",
            "how_to_verify": (
                "python -c \""
                "from src.agentic_ml.security.manifest import ArtifactBundleManager; "
                f"r = ArtifactBundleManager.verify_bundle('{version_dir}'); "
                "print(r)\""
            ),
            "artifacts": list(file_hashes.keys()) + ["manifest.json", "signature.sig"],
            "onnx_available": onnx_exported,
        }
        readme_path = version_dir / "README.json"
        readme_path.write_text(json.dumps(readme, indent=2), encoding="utf-8")

        logger.info(
            "Artifact Bundle created → %s [v=%s, sha256(model)=%s..., algo=Ed25519]",
            version_dir, version_dir.name, file_hashes["model.pkl"][:12],
        )

        return {
            "bundle_dir": str(version_dir),
            "model_path": str(model_path),
            "manifest_path": str(manifest_path),
            "signature_path": str(sig_path),
            "version": version_dir.name,
            "hashes": file_hashes,
            "manifest": manifest_data,
            "onnx_exported": onnx_exported,
        }

    @classmethod
    def verify_bundle(
        cls,
        bundle_dir: str,
        public_key_pem: Optional[bytes] = None,
    ) -> Dict[str, Any]:
        """
        Verify bundle integrity and cryptographic authenticity.

        1. Integrity: Verifies SHA-256 of all constituent files against manifest.json.
        2. Authenticity: Verifies Ed25519 signature in signature.sig over manifest.json.

        Returns dict with:
            valid, integrity_ok, signature_ok, errors, manifest
        """
        b_path = Path(bundle_dir)
        manifest_file = b_path / "manifest.json"
        sig_file = b_path / "signature.sig"

        errors: List[str] = []
        if not manifest_file.exists():
            return {
                "valid": False, "integrity_ok": False, "signature_ok": False,
                "errors": ["manifest.json missing"],
            }
        if not sig_file.exists():
            return {
                "valid": False, "integrity_ok": False, "signature_ok": False,
                "errors": ["signature.sig missing"],
            }

        manifest_bytes = manifest_file.read_bytes()
        try:
            manifest_data = json.loads(manifest_bytes.decode("utf-8"))
        except Exception as exc:
            return {
                "valid": False, "integrity_ok": False, "signature_ok": False,
                "errors": [f"Malformed manifest: {exc}"],
            }

        # ── 1. Integrity: verify SHA-256 of all constituent files ──────────
        integrity_ok = True
        for filename, expected_hash in manifest_data.get("files", {}).items():
            target_file = b_path / filename
            if not target_file.exists():
                integrity_ok = False
                errors.append(f"Missing bundle component: {filename}")
                continue
            actual_hash = compute_sha256(str(target_file))
            if actual_hash != expected_hash:
                integrity_ok = False
                errors.append(
                    f"INTEGRITY FAILURE: Hash mismatch in {filename} (expected {expected_hash[:16]}..., got {actual_hash[:16]}...)"
                )

        # ── 2. Authenticity: verify Ed25519 signature ─────────────────────
        if not public_key_pem:
            _, pub_pem = get_or_create_signing_keys()
        else:
            pub_pem = public_key_pem

        sig_b64 = sig_file.read_text(encoding="utf-8").strip()
        signature_ok = verify_signature(manifest_bytes, sig_b64, pub_pem)
        if not signature_ok:
            errors.append(
                "AUTHENTICITY FAILURE: Digital signature invalid or untrusted public key (Ed25519)."
            )

        valid = integrity_ok and signature_ok

        return {
            "valid": valid,
            "integrity_ok": integrity_ok,
            "signature_ok": signature_ok,
            "errors": errors,
            "manifest": manifest_data,
        }
