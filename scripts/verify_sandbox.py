"""Verify sandbox isolation controls."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agentic_ml.sandbox.models import ExecutionRequest
from src.agentic_ml.sandbox.runner import SandboxRunner

# 1. Timeout enforcement
r1 = SandboxRunner.run(ExecutionRequest(code="import time; time.sleep(600)", timeout_seconds=2))
assert r1.timed_out is True, f"Expected timed_out=True, got {r1.timed_out}"
print(f"TIMEOUT: timed_out={r1.timed_out} [OK]")

# 2. Env variable not leaked
r2 = SandboxRunner.run(ExecutionRequest(code='import os; print(os.environ.get("GROQ_API_KEY", "NOT_FOUND"))', timeout_seconds=5))
assert "NOT_FOUND" in r2.stdout or r2.stdout.strip() == "None", f"Env leaked: {r2.stdout}"
print(f"ENV_ISOLATION: stdout={r2.stdout.strip()!r} [OK]")

# 3. Subprocess blocked
r3 = SandboxRunner.run(ExecutionRequest(code="import subprocess; subprocess.run(['ls'])", timeout_seconds=5))
assert r3.returncode != 0 or "blocked" in r3.stdout.lower() or "blocked" in r3.stderr.lower(), f"subprocess not blocked: {r3.stdout}"
print(f"SUBPROCESS_BLOCK: returncode={r3.returncode} [OK]")


print("RULE_17_SANDBOX_CONTROLS_VERIFIED")
print("KNOWN_LIMITATION: subprocess isolation != OCI container boundary; not claimed as multi-tenant production secure")
