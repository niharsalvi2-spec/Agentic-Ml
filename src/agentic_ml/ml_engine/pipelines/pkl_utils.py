"""
Low-Level PKL Serialization, Integrity Hashing, and Version Registry Utilities.
"""

import os
import sys
import json
import pickle
import hashlib
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Tuple, Dict, List, Optional

try:
    import joblib
    _HAS_JOBLIB = True
except ImportError:
    _HAS_JOBLIB = False


class PKLSecurityError(Exception):
    """Raised when a loaded .pkl file's hash does not match its recorded hash."""


def save_pkl(obj: Any, filepath: str, compress: int = 3, use_joblib: bool = True) -> str:
    """Serializes obj to disk with compression."""
    filepath = str(filepath)
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    if use_joblib and _HAS_JOBLIB:
        joblib.dump(obj, filepath, compress=compress)
    else:
        with open(filepath, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    return filepath


def load_pkl(filepath: str) -> Any:
    """Loads a .pkl file, trying joblib first, falling back to stdlib pickle."""
    filepath = str(filepath)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"PKL file not found: {filepath}")

    if _HAS_JOBLIB:
        try:
            return joblib.load(filepath)
        except Exception:
            pass
    with open(filepath, "rb") as f:
        return pickle.load(f)


def compute_file_hash(filepath: str, algorithm: str = "sha256") -> str:
    """Computes SHA-256 integrity hash of a binary file."""
    h = hashlib.new(algorithm)
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def save_pkl_with_hash(obj: Any, filepath: str, compress: int = 3) -> Tuple[str, str]:
    """Saves .pkl and companion .hash JSON file."""
    saved_path = save_pkl(obj, filepath, compress=compress)
    file_hash = compute_file_hash(saved_path)
    hash_filepath = saved_path + ".hash"
    with open(hash_filepath, "w", encoding="utf-8") as f:
        json.dump({
            "filepath": saved_path,
            "sha256": file_hash,
            "size_bytes": os.path.getsize(saved_path),
            "saved_at": datetime.now().isoformat(),
        }, f, indent=2)
    return saved_path, file_hash


def safe_load_pkl(filepath: str, verify_hash: bool = True) -> Any:
    """Loads a .pkl and validates its hash against the .hash file if present."""
    filepath = str(filepath)
    hash_filepath = filepath + ".hash"

    if verify_hash and os.path.exists(hash_filepath):
        with open(hash_filepath, "r", encoding="utf-8") as f:
            expected = json.load(f)
        actual_hash = compute_file_hash(filepath)
        if actual_hash != expected["sha256"]:
            raise PKLSecurityError(
                f"Hash mismatch for {filepath} — file may have been corrupted or tampered with.\n"
                f"expected: {expected['sha256']}\n"
                f"actual:   {actual_hash}"
            )
    return load_pkl(filepath)


def inspect_pkl(filepath: str, verbose: bool = False) -> Dict[str, Any]:
    """Inspects metadata of a serialized PKL artifact."""
    filepath = str(filepath)
    size_bytes = os.path.getsize(filepath)
    obj = load_pkl(filepath)

    summary: Dict[str, Any] = {
        "filepath": filepath,
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / 1024 / 1024, 3),
        "object_type": type(obj).__name__,
    }

    if isinstance(obj, dict):
        summary["keys"] = list(obj.keys())
    return summary


class PKLVersionManager:
    """Model versioning and registry manager."""

    def __init__(self, registry_dir: str = "models_registry"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(exist_ok=True, parents=True)
        self.registry_file = self.registry_dir / "registry.json"
        self.registry = self._load_registry()

    def _load_registry(self) -> Dict[str, Any]:
        if self.registry_file.exists():
            with open(self.registry_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_registry(self) -> None:
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(self.registry, f, indent=2, default=str)

    def register(self, bundle: Dict[str, Any], model_name: str, metrics: Optional[Dict[str, Any]] = None, description: str = "") -> str:
        model_dir = self.registry_dir / model_name
        model_dir.mkdir(exist_ok=True)

        if model_name in self.registry:
            version_num = len(self.registry[model_name]["versions"]) + 1
        else:
            version_num = 1
            self.registry[model_name] = {"model_name": model_name, "versions": {}, "production": None}

        version_str = f"v{version_num}"
        version_file = model_dir / f"{model_name}_{version_str}.pkl"

        bundle_copy = dict(bundle)
        bundle_copy["version"] = version_str
        bundle_copy["model_name"] = model_name
        bundle_copy["registered_at"] = datetime.now().isoformat()

        save_pkl(bundle_copy, str(version_file))

        self.registry[model_name]["versions"][version_str] = {
            "filepath": str(version_file),
            "metrics": metrics or {},
            "description": description,
            "created_at": bundle_copy["registered_at"],
            "size_bytes": os.path.getsize(version_file),
        }
        self._save_registry()
        return version_str

    def promote_to_production(self, model_name: str, version: str) -> str:
        if model_name not in self.registry:
            raise ValueError(f"Model not found: {model_name}")
        if version not in self.registry[model_name]["versions"]:
            raise ValueError(f"Version not found: {version}")

        self.registry[model_name]["production"] = version
        version_info = self.registry[model_name]["versions"][version]
        production_file = self.registry_dir / model_name / f"{model_name}_production.pkl"

        shutil.copy2(version_info["filepath"], production_file)
        self._save_registry()
        return str(production_file)

    def load_production(self, model_name: str) -> Any:
        production_file = self.registry_dir / model_name / f"{model_name}_production.pkl"
        if not production_file.exists():
            raise FileNotFoundError(f"No production model registered for: {model_name}")
        return load_pkl(str(production_file))
