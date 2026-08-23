"""
RunRegistry — Durable SQLite persistence for pipeline run metadata & HITL approvals.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Dict, Any, Optional, List

from src.agentic_ml.core.constants import ARTIFACTS_DIR

_DB_PATH = ARTIFACTS_DIR / "runtime" / "runs.db"


class RunRegistry:
    """Durable metadata store for run lifecycle, statuses, and idempotent HITL approvals."""

    _instance: Optional[RunRegistry] = None
    _lock = Lock()

    def __init__(self, db_path: Path = _DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @classmethod
    def get(cls) -> RunRegistry:
        with cls._lock:
            if cls._instance is None:
                cls._instance = RunRegistry()
            return cls._instance

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    prompt TEXT NOT NULL,
                    dataset_path TEXT,
                    target_column TEXT,
                    random_seed INTEGER,
                    status TEXT NOT NULL,
                    risk_score INTEGER,
                    risk_level TEXT,
                    deployment_decision TEXT,
                    artifact_path TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS approvals (
                    approval_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    requested_at TEXT NOT NULL,
                    resolved_at TEXT,
                    resolved_by TEXT,
                    FOREIGN KEY (run_id) REFERENCES runs (run_id)
                )
            """)
            conn.commit()

    def create_run(
        self,
        run_id: str,
        prompt: str,
        dataset_path: str = "",
        target_column: Optional[str] = None,
        random_seed: int = 42,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._get_conn() as conn:
            conn.execute("""
                INSERT INTO runs (
                    run_id, prompt, dataset_path, target_column, random_seed,
                    status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'RUNNING', ?, ?)
            """, (run_id, prompt, dataset_path, target_column, random_seed, now, now))
            conn.commit()

    def update_status(
        self,
        run_id: str,
        status: str,
        risk_score: Optional[int] = None,
        risk_level: Optional[str] = None,
        deployment_decision: Optional[str] = None,
        artifact_path: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._get_conn() as conn:
            conn.execute("""
                UPDATE runs
                SET status = ?,
                    risk_score = COALESCE(?, risk_score),
                    risk_level = COALESCE(?, risk_level),
                    deployment_decision = COALESCE(?, deployment_decision),
                    artifact_path = COALESCE(?, artifact_path),
                    error = COALESCE(?, error),
                    updated_at = ?
                WHERE run_id = ?
            """, (status, risk_score, risk_level, deployment_decision, artifact_path, error, now, run_id))
            conn.commit()

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._get_conn() as conn:
            cur = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
            row = cur.fetchone()
            if row:
                return dict(row)
        return None

    def record_hitl_request(self, run_id: str, risk_score: int, risk_level: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        approval_id = f"appr_{run_id}"
        with self._get_conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO approvals (
                    approval_id, run_id, decision, requested_at
                ) VALUES (?, ?, 'PENDING', ?)
            """, (approval_id, run_id, now))
            conn.commit()
        self.update_status(run_id, status="AWAITING_APPROVAL", risk_score=risk_score, risk_level=risk_level)

    def resolve_hitl_approval(self, run_id: str, approved: bool, resolved_by: str = "user") -> bool:
        """
        Idempotently resolve HITL approval.
        Returns True if transition succeeded, False if already resolved.
        """
        now = datetime.now(timezone.utc).isoformat()
        decision_str = "HUMAN_APPROVED" if approved else "REJECTED"
        approval_id = f"appr_{run_id}"
        with self._get_conn() as conn:
            cur = conn.execute("SELECT decision FROM approvals WHERE approval_id = ?", (approval_id,))
            row = cur.fetchone()
            if not row or row["decision"] != "PENDING":
                return False

            conn.execute("""
                UPDATE approvals
                SET decision = ?, resolved_at = ?, resolved_by = ?
                WHERE approval_id = ?
            """, (decision_str, now, resolved_by, approval_id))
            conn.commit()

        self.update_status(run_id, status="RESUMING", deployment_decision=decision_str)
        return True
