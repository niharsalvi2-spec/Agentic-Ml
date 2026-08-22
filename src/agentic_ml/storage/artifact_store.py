"""
ArtifactStore — content-addressed, sha256-tracked artifact persistence.

Design:
  - Every artifact saved gets a sha256 digest logged.
  - Agents receive artifact refs (paths) from state — they never re-preprocess raw data.
  - Directory layout: artifacts/{run_id}/{stage}/{filename}
  - Parquet is used for DataFrames (space-efficient, typed, schema-preserving).
  - joblib is used for sklearn Pipeline/Preprocessor objects.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("agentic_ml.storage.artifact_store")

_ARTIFACTS_ROOT = Path("artifacts")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _ensure_dir(run_id: str, stage: str) -> Path:
    d = _ARTIFACTS_ROOT / run_id / stage
    d.mkdir(parents=True, exist_ok=True)
    return d


class ArtifactStore:
    """
    Save and load typed artifacts with sha256 tracking.

    All save methods return (path_str, sha256_hex) so callers can store
    the ref in AgentState.artifact_refs for downstream agents to consume.
    """

    # ── DataFrame ──────────────────────────────────────────────────────────

    @staticmethod
    def save_dataframe(run_id: str, stage: str, df: Any, filename: str = "data.parquet") -> Tuple[str, str]:
        """Save a pandas DataFrame as Parquet. Returns (path, sha256)."""
        d = _ensure_dir(run_id, stage)
        path = d / filename
        df.to_parquet(str(path), index=False, compression="snappy")
        digest = _sha256_file(path)
        logger.info("ArtifactStore.save_dataframe: %s [sha256:%s...]", path, digest[:12])
        return str(path), digest

    @staticmethod
    def load_dataframe(ref: str) -> Any:
        """Load a Parquet artifact by path."""
        import pandas as pd
        path = Path(ref)
        if not path.exists():
            raise FileNotFoundError(f"ArtifactStore: dataframe ref not found: {ref}")
        return pd.read_parquet(str(path))

    # ── Preprocessor ───────────────────────────────────────────────────────

    @staticmethod
    def save_preprocessor(run_id: str, preprocessor: Any) -> Tuple[str, str]:
        """Save a fitted DeterministicPreprocessor via joblib. Returns (path, sha256)."""
        import joblib
        d = _ensure_dir(run_id, "preprocessing")
        path = d / "preprocessor.pkl"
        joblib.dump(preprocessor, str(path), compress=3)
        digest = _sha256_file(path)
        logger.info("ArtifactStore.save_preprocessor: %s [sha256:%s...]", path, digest[:12])
        return str(path), digest

    @staticmethod
    def load_preprocessor(ref: str) -> Any:
        """Load a fitted preprocessor by path."""
        import joblib
        path = Path(ref)
        if not path.exists():
            raise FileNotFoundError(f"ArtifactStore: preprocessor ref not found: {ref}")
        return joblib.load(str(path))

    # ── JSON metadata ──────────────────────────────────────────────────────

    @staticmethod
    def save_json(run_id: str, stage: str, data: Dict, filename: str) -> Tuple[str, str]:
        """Save a dict as JSON artifact. Returns (path, sha256)."""
        d = _ensure_dir(run_id, stage)
        path = d / filename
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        digest = _sha256_file(path)
        logger.info("ArtifactStore.save_json: %s [sha256:%s...]", path, digest[:12])
        return str(path), digest

    @staticmethod
    def load_json(ref: str) -> Dict:
        """Load a JSON artifact by path."""
        path = Path(ref)
        if not path.exists():
            raise FileNotFoundError(f"ArtifactStore: json ref not found: {ref}")
        return json.loads(path.read_text(encoding="utf-8"))

    # ── Provenance linker ──────────────────────────────────────────────────

    @staticmethod
    def write_stage_provenance(
        run_id: str,
        stage: str,
        operation: str,
        inputs: Dict[str, str],   # name → sha256
        outputs: Dict[str, str],  # name → sha256
        agent_name: str,
        metadata: Optional[Dict] = None,
    ) -> str:
        """Write a provenance record linking inputs to outputs. Returns path."""
        record = {
            "run_id": run_id,
            "stage": stage,
            "agent": agent_name,
            "operation": operation,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "inputs": inputs,
            "outputs": outputs,
            "metadata": metadata or {},
        }
        path, _ = ArtifactStore.save_json(run_id, stage, record, "stage_provenance.json")
        return path
