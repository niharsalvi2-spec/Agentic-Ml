"""
AGENT-READABLE MODULE
======================
name: pkl_utils
purpose: Low-level, dependency-light utilities for serializing ANY trained
         object (a bare model, a full sklearn Pipeline, or a bundle dict) to
         a .pkl file, loading it back, inspecting its contents, verifying its
         integrity, and managing multiple versions. This is the layer
         `pkl_generator_agent.py` builds on.

FUNCTIONS
---------
save_pkl(obj, filepath, compress=3, use_joblib=True) -> filepath
load_pkl(filepath) -> obj                              # tries joblib, falls back to pickle
compute_file_hash(filepath, algorithm="sha256") -> hex string
save_pkl_with_hash(obj, filepath, compress=3) -> (filepath, hash)
safe_load_pkl(filepath, verify_hash=True) -> obj        # raises if hash file exists and mismatches
inspect_pkl(filepath, verbose=True) -> dict summary of what's inside

CLASSES
-------
PKLVersionManager — simple file-based model registry: register a version,
    promote one to "production", load production or a specific version,
    compare metrics across versions.

WHY joblib OVER pickle FOR ML MODELS
-------------------------------------
joblib is preferred for sklearn-style objects because it is more efficient
for numpy-array-heavy objects (tree ensembles, coefficient matrices), supports
compression, and supports memory-mapped loading for very large models. This
module defaults to joblib and transparently falls back to stdlib pickle only
if joblib is unavailable.

SECURITY WARNING
-----------------
Pickle/joblib files can execute arbitrary code on load. NEVER load a .pkl
file from an untrusted source. `save_pkl_with_hash` + `safe_load_pkl` guard
against silent tampering of files YOU produced — they do not make it safe to
load a file from someone else.
"""

import os
import sys
import json
import pickle
import hashlib
from datetime import datetime
from pathlib import Path

try:
    import joblib
    _HAS_JOBLIB = True
except ImportError:
    _HAS_JOBLIB = False


class PKLSecurityError(Exception):
    """Raised when a loaded .pkl file's hash does not match its recorded hash."""


# --------------------------------------------------------------------------- #
# Save / Load
# --------------------------------------------------------------------------- #
def save_pkl(obj, filepath, compress=3, use_joblib=True):
    """
    Serialize `obj` (a model, a sklearn Pipeline, or any bundle dict) to disk.
    compress: 0 (fastest, largest) .. 9 (slowest, smallest). Ignored if
    use_joblib=False (stdlib pickle has no built-in compression level).
    """
    filepath = str(filepath)
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    if use_joblib and _HAS_JOBLIB:
        joblib.dump(obj, filepath, compress=compress)
    else:
        with open(filepath, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    return filepath


def load_pkl(filepath):
    """Load a .pkl file, trying joblib first (handles both joblib- and
    pickle-saved files), falling back to stdlib pickle."""
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


# --------------------------------------------------------------------------- #
# Hashing / integrity
# --------------------------------------------------------------------------- #
def compute_file_hash(filepath, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def save_pkl_with_hash(obj, filepath, compress=3):
    """Save the .pkl and a companion `<filepath>.hash` JSON file recording its
    sha256, so tampering can be detected later by `safe_load_pkl`."""
    filepath = save_pkl(obj, filepath, compress=compress)
    file_hash = compute_file_hash(filepath)
    hash_filepath = filepath + ".hash"
    with open(hash_filepath, "w") as f:
        json.dump({
            "filepath": filepath,
            "sha256": file_hash,
            "size_bytes": os.path.getsize(filepath),
            "saved_at": datetime.now().isoformat(),
        }, f, indent=2)
    return filepath, file_hash


def safe_load_pkl(filepath, verify_hash=True):
    """Load a .pkl, verifying its companion .hash file if present and
    verify_hash=True. Raises PKLSecurityError on mismatch."""
    filepath = str(filepath)
    hash_filepath = filepath + ".hash"

    if verify_hash and os.path.exists(hash_filepath):
        with open(hash_filepath) as f:
            expected = json.load(f)
        actual_hash = compute_file_hash(filepath)
        if actual_hash != expected["sha256"]:
            raise PKLSecurityError(
                f"Hash mismatch for {filepath} — file may have been tampered with.\n"
                f"expected: {expected['sha256']}\n"
                f"actual:   {actual_hash}"
            )
    return load_pkl(filepath)


# --------------------------------------------------------------------------- #
# Inspection
# --------------------------------------------------------------------------- #
def inspect_pkl(filepath, verbose=True):
    """
    Load a .pkl and return a summary dict describing its contents without
    requiring the caller to already know its structure. Works for a bare
    model, a sklearn Pipeline, or a bundle dict.
    """
    filepath = str(filepath)
    size_bytes = os.path.getsize(filepath)
    obj = load_pkl(filepath)

    summary = {
        "filepath": filepath,
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / 1024 / 1024, 3),
        "object_type": type(obj).__name__,
        "memory_bytes": sys.getsizeof(obj),
    }

    if isinstance(obj, dict):
        keys_summary = {}
        for key, value in obj.items():
            if isinstance(value, (list, tuple)):
                keys_summary[key] = f"{type(value).__name__}(len={len(value)})"
            elif isinstance(value, dict):
                keys_summary[key] = f"dict(keys={list(value.keys())[:5]})"
            elif isinstance(value, float):
                keys_summary[key] = f"float={value:.4f}"
            elif isinstance(value, str) and len(value) > 60:
                keys_summary[key] = f"str='{value[:60]}...'"
            else:
                keys_summary[key] = f"{type(value).__name__}={value}"
        summary["keys"] = keys_summary
    elif hasattr(obj, "get_params"):
        summary["params"] = obj.get_params()

    if verbose:
        print(f"=== {filepath} ===")
        print(f"size: {summary['size_mb']} MB | type: {summary['object_type']}")
        if "keys" in summary:
            for k, v in summary["keys"].items():
                print(f"  '{k}': {v}")
        elif "params" in summary:
            for k, v in list(summary["params"].items())[:15]:
                print(f"  {k}: {v}")

    return summary


# --------------------------------------------------------------------------- #
# Version management (simple file-based model registry)
# --------------------------------------------------------------------------- #
class PKLVersionManager:
    """
    A minimal model registry: register new .pkl versions under a model name,
    promote one to "production", load production or a specific version, and
    compare metrics across all registered versions.
    """

    def __init__(self, registry_dir="model_registry"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(exist_ok=True, parents=True)
        self.registry_file = self.registry_dir / "registry.json"
        self.registry = self._load_registry()

    def _load_registry(self):
        if self.registry_file.exists():
            with open(self.registry_file) as f:
                return json.load(f)
        return {}

    def _save_registry(self):
        with open(self.registry_file, "w") as f:
            json.dump(self.registry, f, indent=2, default=str)

    def register(self, bundle, model_name, metrics=None, description="", compress=3):
        """Save `bundle` (typically the output of PKLGeneratorAgent.build_bundle)
        as the next version of `model_name` and record it in the registry."""
        model_dir = self.registry_dir / model_name
        model_dir.mkdir(exist_ok=True)

        if model_name in self.registry:
            version_num = len(self.registry[model_name]["versions"]) + 1
        else:
            version_num = 1
            self.registry[model_name] = {"model_name": model_name, "versions": {}, "production": None}

        version_str = f"v{version_num}"
        version_file = model_dir / f"{model_name}_{version_str}.pkl"

        bundle = dict(bundle)  # shallow copy so we don't mutate caller's dict
        bundle["version"] = version_str
        bundle["model_name"] = model_name
        bundle["registered_at"] = datetime.now().isoformat()

        save_pkl(bundle, version_file, compress=compress)

        self.registry[model_name]["versions"][version_str] = {
            "filepath": str(version_file),
            "metrics": metrics or {},
            "description": description,
            "created_at": bundle["registered_at"],
            "size_bytes": os.path.getsize(version_file),
        }
        self._save_registry()
        return version_str

    def promote_to_production(self, model_name, version):
        if model_name not in self.registry:
            raise ValueError(f"Model not found: {model_name}")
        if version not in self.registry[model_name]["versions"]:
            raise ValueError(f"Version not found: {version}")

        self.registry[model_name]["production"] = version
        version_info = self.registry[model_name]["versions"][version]
        production_file = self.registry_dir / model_name / f"{model_name}_production.pkl"

        import shutil
        shutil.copy2(version_info["filepath"], production_file)
        self._save_registry()
        return str(production_file)

    def load_production(self, model_name):
        production_file = self.registry_dir / model_name / f"{model_name}_production.pkl"
        if not production_file.exists():
            raise FileNotFoundError(f"No production model registered for: {model_name}")
        return load_pkl(production_file)

    def load_version(self, model_name, version):
        version_info = self.registry[model_name]["versions"][version]
        return load_pkl(version_info["filepath"])

    def compare_versions(self, model_name, verbose=True):
        if model_name not in self.registry:
            raise ValueError(f"Model not found: {model_name}")

        production = self.registry[model_name]["production"]
        rows = []
        for ver, info in self.registry[model_name]["versions"].items():
            rows.append({
                "version": ver,
                "metrics": info["metrics"],
                "size_kb": round(info["size_bytes"] / 1024, 1),
                "created_at": info["created_at"][:10],
                "is_production": ver == production,
            })

        if verbose:
            print(f"=== Version history: {model_name} ===")
            for r in rows:
                tag = " [PRODUCTION]" if r["is_production"] else ""
                print(f"  {r['version']}: {r['metrics']} | {r['size_kb']}KB | {r['created_at']}{tag}")
        return rows
