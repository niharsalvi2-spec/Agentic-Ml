"""
Path Sanitizer — Strictly prevents directory traversal and arbitrary file read/write attacks.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from src.agentic_ml.core.constants import ROOT_DIR, DATA_DIR, ARTIFACTS_DIR

_ALLOWED_ROOTS = [ROOT_DIR.resolve(), DATA_DIR.resolve(), ARTIFACTS_DIR.resolve()]


def sanitize_dataset_path(raw_path: str) -> Optional[Path]:
    """
    Validate and sanitize dataset path.
    Rejects directory traversal (../), null bytes, and paths escaping allowed directories.
    Returns resolved Path or raises ValueError.
    """
    if not raw_path or not raw_path.strip():
        return None

    if "\0" in raw_path:
        raise ValueError("Null bytes are forbidden in paths.")

    # Check for obvious traversal sequences
    normalized = os.path.normpath(raw_path)
    if ".." in normalized.split(os.sep):
        raise ValueError(f"Path traversal detected in path: {raw_path}")

    path_obj = Path(raw_path)
    if not path_obj.is_absolute():
        # Relative to DATA_DIR or ROOT_DIR
        resolved = (DATA_DIR / raw_path).resolve()
        if not resolved.exists():
            resolved = (ROOT_DIR / raw_path).resolve()
    else:
        resolved = path_obj.resolve()

    # Verify path is within allowed directories
    is_allowed = any(resolved == root or root in resolved.parents for root in _ALLOWED_ROOTS)
    if not is_allowed:
        raise ValueError(f"Path '{raw_path}' escapes allowed directory boundaries.")

    return resolved
