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

    def test_invalid_checkpoint_backend_fails_fast(self):
        with patch.dict(os.environ, {"CHECKPOINT_BACKEND": "redis_unsupported"}):
            reset_checkpointer()
            with pytest.raises(ValueError) as exc_info:
                get_checkpointer()
            assert "Invalid CHECKPOINT_BACKEND" in str(exc_info.value)
