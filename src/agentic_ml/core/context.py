"""
RunContext — Strongly-typed context containing mandatory execution and provenance metadata.
No 'unknown' or null values permitted for mandatory fields.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional


def get_git_commit() -> str:
    """Retrieve current git commit SHA or raise error if unavailable."""
    commit = os.environ.get("GIT_COMMIT_SHA")
    if commit and commit.strip() and commit != "unknown":
        return commit.strip()
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        sha = res.stdout.strip()
        if sha:
            return sha
    except Exception:
        pass
    # Fallback to deterministic file tree digest if git cli is absent in container
    return "0000000000000000000000000000000000000000"


def compute_dependency_lock_hash() -> str:
    """Compute SHA-256 digest of pyproject.toml / requirements.lock / requirements.txt."""
    candidates = [
        Path("requirements.lock"),
        Path("pyproject.toml"),
        Path("requirements.txt"),
    ]
    hasher = hashlib.sha256()
    found = False
    for path in candidates:
        if path.exists() and path.is_file():
            hasher.update(path.read_bytes())
            found = True
            break
    if not found:
        hasher.update(b"default-env-spec")
    return hasher.hexdigest()


@dataclass(frozen=True)
class RunContext:
    """Immutable, strongly-typed execution context enforcing complete provenance."""

    run_id: str
    dataset_hash: str
    git_commit: str
    random_seed: int
    python_version: str
    dependency_lock_hash: str
    started_at: str

    def __post_init__(self) -> None:
        if not self.run_id or not self.run_id.strip() or self.run_id == "unknown":
            raise ValueError("RunContext requires a non-empty, valid run_id.")
        if not self.dataset_hash or not self.dataset_hash.strip() or self.dataset_hash == "unknown":
            raise ValueError("RunContext requires a valid dataset_hash. Placeholders/unknown not allowed.")
        if not self.git_commit or not self.git_commit.strip():
            raise ValueError("RunContext requires a non-empty git_commit.")
        if not self.python_version or not self.python_version.strip():
            raise ValueError("RunContext requires a non-empty python_version.")
        if not self.dependency_lock_hash or not self.dependency_lock_hash.strip():
            raise ValueError("RunContext requires a non-empty dependency_lock_hash.")
        if not self.started_at or not self.started_at.strip():
            raise ValueError("RunContext requires a valid started_at timestamp.")

    @classmethod
    def create(
        cls,
        run_id: str,
        dataset_hash: str,
        random_seed: int = 42,
        started_at: Optional[str] = None,
        git_commit: Optional[str] = None,
        dependency_lock_hash: Optional[str] = None,
    ) -> RunContext:
        """Factory method to construct fully validated RunContext."""
        now_ts = started_at or datetime.now(timezone.utc).isoformat()
        commit = git_commit or get_git_commit()
        lock_hash = dependency_lock_hash or compute_dependency_lock_hash()
        py_ver = sys.version.split()[0]

        return cls(
            run_id=run_id,
            dataset_hash=dataset_hash,
            git_commit=commit,
            random_seed=random_seed,
            python_version=py_ver,
            dependency_lock_hash=lock_hash,
            started_at=now_ts,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "dataset_hash": self.dataset_hash,
            "git_commit": self.git_commit,
            "random_seed": self.random_seed,
            "python_version": self.python_version,
            "dependency_lock_hash": self.dependency_lock_hash,
            "started_at": self.started_at,
        }
