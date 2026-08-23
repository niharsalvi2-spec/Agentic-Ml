import os
import sys
import concurrent.futures
import tempfile
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agentic_ml.storage.run_registry import RunRegistry


def main():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "hitl_concurrency_audit.db"
        RunRegistry.reset()
        reg = RunRegistry.get(db_path)

        # 1. 50 concurrent approval attempts
        run_id = "run_audit_concurrency_50"
        reg.create_run(run_id, prompt="50 thread race test")
        reg.record_hitl_request(run_id, risk_score=85, risk_level="HIGH")

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            futures = [
                executor.submit(reg.resolve_hitl_approval, run_id, approved=True, resolved_by=f"user_{i}")
                for i in range(50)
            ]
            for f in concurrent.futures.as_completed(futures):
                results.append(f.result())

        assert results.count(True) == 1, f"Expected 1 True, got {results.count(True)}"
        assert results.count(False) == 49, f"Expected 49 False, got {results.count(False)}"
        print("TEST 1 (50-thread concurrent CAS): PASSED (1 True, 49 False)")

        # 2. Approve vs Reject race
        run_id_race = "run_audit_race_appr_rej"
        reg.create_run(run_id_race, prompt="Approve vs reject race")
        reg.record_hitl_request(run_id_race, risk_score=90, risk_level="HIGH")

        race_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [
                executor.submit(reg.resolve_hitl_approval, run_id_race, approved=(i % 2 == 0), resolved_by=f"user_{i}")
                for i in range(20)
            ]
            for f in concurrent.futures.as_completed(futures):
                race_results.append(f.result())

        assert race_results.count(True) == 1, f"Expected 1 True, got {race_results.count(True)}"
        assert race_results.count(False) == 19, f"Expected 19 False, got {race_results.count(False)}"
        print("TEST 2 (Approve vs Reject Race): PASSED (1 True, 19 False)")

        print("RULE_6_HITL_CONCURRENCY_VERIFIED_SUCCESS")

if __name__ == "__main__":
    main()
