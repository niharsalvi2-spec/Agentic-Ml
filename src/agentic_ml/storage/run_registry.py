"""
RunRegistry — Durable SQLite persistence for pipeline run metadata & atomic HITL approvals.
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
    """Durable metadata store for run lifecycle, statuses, and atomic compare-and-set HITL approvals."""

    _instance: Optional[RunRegistry] = None
    _lock = Lock()

    def __init__(self, db_path: Path = _DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @classmethod
    def get(cls, db_path: Optional[Path] = None) -> RunRegistry:
        with cls._lock:
            if cls._instance is None:
                cls._instance = RunRegistry(db_path or _DB_PATH)
            return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset singleton instance (useful for test isolation)."""
        with cls._lock:
            cls._instance = None

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=30.0)
        conn.row_factory = sqlite3.Row
        # Enable WAL mode for concurrency
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
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
                    last_sequence_number INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS approvals (
                    approval_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL UNIQUE,
                    decision TEXT NOT NULL,
                    requested_at TEXT NOT NULL,
                    resolved_at TEXT,
                    resolved_by TEXT,
                    FOREIGN KEY (run_id) REFERENCES runs (run_id)
                )
            """)
            # Auto-migrate existing runs table if needed
            cur = conn.cursor()
            cur.execute("PRAGMA table_info(runs);")
            columns = [row["name"] for row in cur.fetchall()]
            if "last_sequence_number" not in columns and len(columns) > 0:
                conn.execute("ALTER TABLE runs ADD COLUMN last_sequence_number INTEGER DEFAULT 0;")

            conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);")
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
                    status, last_sequence_number, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'RUNNING', 0, ?, ?)
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
        last_sequence_number: Optional[int] = None,
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
                    last_sequence_number = COALESCE(?, last_sequence_number),
                    updated_at = ?
                WHERE run_id = ?
            """, (status, risk_score, risk_level, deployment_decision, artifact_path, error, last_sequence_number, now, run_id))
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
                INSERT INTO approvals (approval_id, run_id, decision, requested_at)
                VALUES (?, ?, 'PENDING', ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    decision = 'PENDING',
                    requested_at = excluded.requested_at,
                    resolved_at = NULL,
                    resolved_by = NULL
            """, (approval_id, run_id, now))
            conn.execute("""
                UPDATE runs
                SET status = 'AWAITING_APPROVAL',
                    risk_score = ?,
                    risk_level = ?,
                    updated_at = ?
                WHERE run_id = ?
            """, (risk_score, risk_level, now, run_id))
            conn.commit()

    def resolve_hitl_approval(self, run_id: str, approved: bool, resolved_by: str = "user") -> bool:
        """
        Atomically resolve HITL approval using Compare-And-Set (CAS) semantics.
        Guarantees exactly-once execution: only one concurrent call can transition
        the run from AWAITING_APPROVAL / PENDING to RESUMING / resolved.

        Returns True if transition succeeded, False if already resolved or invalid state.
        """
        now = datetime.now(timezone.utc).isoformat()
        decision_str = "HUMAN_APPROVED" if approved else "REJECTED"
        next_status = "RESUMING" if approved else "REJECTED"
        approval_id = f"appr_{run_id}"

        with self._get_conn() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            try:
                # 1. Atomic compare-and-set on approvals table
                cur_appr = conn.execute("""
                    UPDATE approvals
                    SET decision = ?,
                        resolved_at = ?,
                        resolved_by = ?
                    WHERE approval_id = ? AND decision = 'PENDING'
                """, (decision_str, now, resolved_by, approval_id))

                if cur_appr.rowcount != 1:
                    conn.rollback()
                    return False

                # 2. Atomic compare-and-set on runs table
                cur_runs = conn.execute("""
                    UPDATE runs
                    SET status = ?,
                        deployment_decision = ?,
                        updated_at = ?
                    WHERE run_id = ? AND status = 'AWAITING_APPROVAL'
                """, (next_status, decision_str, now, run_id))

                if cur_runs.rowcount != 1:
                    conn.rollback()
                    return False

                conn.commit()
                return True
            except Exception:
                conn.rollback()
                raise
