"""
Tests for Atomic HITL Compare-And-Set Transitions and Concurrency (Phase 2).
"""
import concurrent.futures
import pytest
from pathlib import Path

from src.agentic_ml.storage.run_registry import RunRegistry


class TestHITLAtomicTransitions:

    @pytest.fixture
    def registry(self, tmp_path):
        db_path = tmp_path / "test_hitl.db"
        RunRegistry.reset()
        reg = RunRegistry.get(db_path)
        return reg

    def test_single_approval_success(self, registry):
        run_id = "run_test_hitl_001"
        registry.create_run(run_id, prompt="Test prompt")
        registry.record_hitl_request(run_id, risk_score=75, risk_level="HIGH")

        run = registry.get_run(run_id)
        assert run["status"] == "AWAITING_APPROVAL"

        # First resolution succeeds
        success = registry.resolve_hitl_approval(run_id, approved=True, resolved_by="test_reviewer")
        assert success is True

        updated_run = registry.get_run(run_id)
        assert updated_run["status"] == "RESUMING"
        assert updated_run["deployment_decision"] == "HUMAN_APPROVED"

    def test_duplicate_approval_rejected_exactly_once(self, registry):
        run_id = "run_test_hitl_002"
        registry.create_run(run_id, prompt="Test prompt")
        registry.record_hitl_request(run_id, risk_score=80, risk_level="HIGH")

        # First approval
        res1 = registry.resolve_hitl_approval(run_id, approved=True)
        assert res1 is True

        # Duplicate approval must fail atomic transition
        res2 = registry.resolve_hitl_approval(run_id, approved=True)
        assert res2 is False

    def test_duplicate_rejection_rejected_exactly_once(self, registry):
        run_id = "run_test_hitl_003"
        registry.create_run(run_id, prompt="Test prompt")
        registry.record_hitl_request(run_id, risk_score=85, risk_level="HIGH")

        # First rejection
        res1 = registry.resolve_hitl_approval(run_id, approved=False)
        assert res1 is True

        # Second rejection must fail
        res2 = registry.resolve_hitl_approval(run_id, approved=False)
        assert res2 is False

        updated_run = registry.get_run(run_id)
        assert updated_run["status"] == "REJECTED"
        assert updated_run["deployment_decision"] == "REJECTED"

    def test_concurrent_simultaneous_approval_requests_race(self, registry):
        run_id = "run_test_hitl_concurrent"
        registry.create_run(run_id, prompt="Test prompt")
        registry.record_hitl_request(run_id, risk_score=90, risk_level="HIGH")

        results = []

        def worker(approved: bool):
            return registry.resolve_hitl_approval(run_id, approved=approved)

        # Launch 10 simultaneous approval/rejection attempts
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(worker, i % 2 == 0)
                for i in range(10)
            ]
            for f in concurrent.futures.as_completed(futures):
                results.append(f.result())

        # Exactly ONE call must succeed (True), all other 9 must fail (False)
        assert results.count(True) == 1
        assert results.count(False) == 9
