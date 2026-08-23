"""
Tests for Checkpointing Architecture, Persistence, and Backend Failures (Phase 1).
"""
import os
import pytest
import sqlite3
from pathlib import Path
from unittest.mock import patch

from src.agentic_ml.api.run_manager import get_checkpointer, reset_checkpointer, stream_run, resume_run
from src.agentic_ml.storage.run_registry import RunRegistry
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import MemorySaver


class TestCheckpointArchitecture:

    def setup_method(self):
        reset_checkpointer()

    def teardown_method(self):
        reset_checkpointer()

    def test_default_sqlite_checkpointer_creation(self, tmp_path):
        db_file = tmp_path / "runs.db"
        with patch.dict(os.environ, {"CHECKPOINT_BACKEND": "sqlite", "CHECKPOINT_DB_PATH": str(db_file)}):
            reset_checkpointer()
            checkpointer = get_checkpointer()
            assert isinstance(checkpointer, SqliteSaver)
            assert db_file.exists()

    def test_explicit_memory_checkpointer_allowed_only_in_dev_mode(self):
        with patch.dict(os.environ, {"CHECKPOINT_BACKEND": "memory"}):
            reset_checkpointer()
            checkpointer = get_checkpointer()
            assert isinstance(checkpointer, MemorySaver)

    def test_sqlite_failure_raises_explicit_error_without_silent_memory_fallback(self):
        # Point to an invalid read-only/uncreatable directory or mock connection failure
        with patch.dict(os.environ, {"CHECKPOINT_BACKEND": "sqlite", "CHECKPOINT_DB_PATH": "Z:\\nonexistent_dir\\runs.db"}):
            reset_checkpointer()
            with pytest.raises(RuntimeError) as exc_info:
                get_checkpointer()
            assert "Silent fallback to memory is forbidden" in str(exc_info.value)

    def test_checkpoint_persistence_across_process_restart(self, tmp_path):
        db_file = tmp_path / "runs.db"
        with patch.dict(os.environ, {"CHECKPOINT_BACKEND": "sqlite", "CHECKPOINT_DB_PATH": str(db_file)}):
            reset_checkpointer()
            checkpointer1 = get_checkpointer()
            config = {"configurable": {"thread_id": "run_test_restart_001", "checkpoint_ns": ""}}
            checkpoint_data = {
                "v": 1,
                "ts": "2026-08-23T00:00:00Z",
                "id": "chk_001",
                "channel_values": {"messages": ["hello"], "status": "in_progress"},
                "channel_versions": {"messages": 1, "status": 1},
                "versions_seen": {},
            }
            metadata = {"source": "test", "step": 1, "writes": {}}
            checkpointer1.put(config, checkpoint_data, metadata, {})

            # Simulate process exit and restart
            reset_checkpointer()
            checkpointer2 = get_checkpointer()
            retrieved = checkpointer2.get(config)
            assert retrieved is not None
            assert retrieved["channel_values"]["status"] == "in_progress"
            assert retrieved["channel_values"]["messages"] == ["hello"]

    def test_invalid_checkpoint_backend_fails_fast(self):
        with patch.dict(os.environ, {"CHECKPOINT_BACKEND": "redis_unsupported"}):
            reset_checkpointer()
            with pytest.raises(ValueError) as exc_info:
                get_checkpointer()
            assert "Invalid CHECKPOINT_BACKEND" in str(exc_info.value)

    def test_resume_missing_run_fails_gracefully(self, tmp_path):
        import asyncio
        db_path = tmp_path / "registry.db"
        RunRegistry.reset()
        RunRegistry.get(db_path)

        async def _run():
            evts = []
            async for sse_event in resume_run("non_existent_run_999", approved=True):
                evts.append(sse_event)
            return evts

        events = asyncio.run(_run())
        assert len(events) >= 1
        assert "run_failed" in events[0].lower() or "not found" in events[0].lower()

    def test_ml_checkpoint_serializer_round_trip(self):
        import pandas as pd
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        from src.agentic_ml.api.run_manager import MLCheckpointSerializer

        serializer = MLCheckpointSerializer()

        # 1. Primitives / JSON types
        prim = {"run_id": "run_123", "score": 0.95, "features": ["f1", "f2"]}
        typed_prim = serializer.dumps_typed(prim)
        loaded_prim = serializer.loads_typed(typed_prim)
        assert loaded_prim == prim

        # 2. Pandas DataFrame / Numpy array
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        typed_df = serializer.dumps_typed(df)
        loaded_df = serializer.loads_typed(typed_df)
        pd.testing.assert_frame_equal(df, loaded_df)

        # 3. Sklearn Model
        rf = RandomForestClassifier(n_estimators=5, random_state=42)
        rf.fit([[1, 2], [3, 4]], [0, 1])
        typed_model = serializer.dumps_typed(rf)
        loaded_model = serializer.loads_typed(typed_model)
        assert isinstance(loaded_model, RandomForestClassifier)
        assert loaded_model.n_estimators == 5





