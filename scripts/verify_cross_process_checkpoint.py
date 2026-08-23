"""
Verification script: Cross-process SQLite Checkpoint Persistence Test.
Spawns two separate OS processes:
  - Process A: writes checkpoint to SQLite database, then terminates.
  - Process B: opens the SQLite database, verifies checkpoint retrieval and state restoration.
"""
import os
import sys
import subprocess
import tempfile
from pathlib import Path

def main():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "cross_process_test.db"

        # Code for Process A
        proc_a_code = f"""
import os, sys
from pathlib import Path
from src.agentic_ml.api.run_manager import get_checkpointer, reset_checkpointer

os.environ["CHECKPOINT_BACKEND"] = "sqlite"
os.environ["CHECKPOINT_DB_PATH"] = r"{db_path}"

reset_checkpointer()
checkpointer = get_checkpointer()

config = {{"configurable": {{"thread_id": "thread_cross_proc_001", "checkpoint_ns": ""}}}}
checkpoint = {{
    "v": 1,
    "ts": "2026-08-23T12:00:00Z",
    "id": "chk_proc_a",
    "channel_values": {{
        "messages": ["message_from_process_a"],
        "task_type": "classification",
        "best_model_name": "RandomForest",
        "metrics": {{"accuracy": 0.95}}
    }},
    "channel_versions": {{"messages": 1, "task_type": 1, "best_model_name": 1, "metrics": 1}},
    "versions_seen": {{}},
}}
metadata = {{"source": "process_a", "step": 1, "writes": {{}}}}

checkpointer.put(config, checkpoint, metadata, {{}})
reset_checkpointer()
print("PROCESS_A_SUCCESS")
"""

        # Code for Process B
        proc_b_code = f"""
import os, sys
from pathlib import Path
from src.agentic_ml.api.run_manager import get_checkpointer, reset_checkpointer

os.environ["CHECKPOINT_BACKEND"] = "sqlite"
os.environ["CHECKPOINT_DB_PATH"] = r"{db_path}"

reset_checkpointer()
checkpointer = get_checkpointer()

config = {{"configurable": {{"thread_id": "thread_cross_proc_001", "checkpoint_ns": ""}}}}
retrieved = checkpointer.get(config)

if retrieved is None:
    print("FAILED: retrieved checkpoint is None")
    sys.exit(1)

values = retrieved.get("channel_values", {{}})
if values.get("best_model_name") != "RandomForest":
    print(f"FAILED: best_model_name mismatch: {{values}}")
    sys.exit(1)

if values.get("metrics", {{}}).get("accuracy") != 0.95:
    print(f"FAILED: metrics mismatch: {{values}}")
    sys.exit(1)

print("PROCESS_B_SUCCESS")
"""

        # Execute Process A
        res_a = subprocess.run(
            [sys.executable, "-c", proc_a_code],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if "PROCESS_A_SUCCESS" not in res_a.stdout:
            print(f"Process A failed: {res_a.stderr} | {res_a.stdout}")
            sys.exit(1)

        # Execute Process B
        res_b = subprocess.run(
            [sys.executable, "-c", proc_b_code],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if "PROCESS_B_SUCCESS" not in res_b.stdout:
            print(f"Process B failed: {res_b.stderr} | {res_b.stdout}")
            sys.exit(1)

        print("CROSS_PROCESS_PERSISTENCE_VERIFIED_SUCCESS")

if __name__ == "__main__":
    main()
