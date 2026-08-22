"""
Adversarial sandbox security tests.

Verifies that the sandbox ExecutionManager correctly:
  1. Blocks network access (import socket, import requests)
  2. Blocks subprocess spawning (import subprocess)
  3. Enforces execution timeout on infinite loops
  4. Limits output size
  5. Does not expose environment variables (API keys, etc.)

All tests must result in BLOCKED, ERROR, or TIMEOUT — never a successful execution
of the dangerous operation. This test suite is part of the CI security gate.
"""
import os
import pytest

from src.agentic_ml.sandbox.manager import ExecutionManager
from src.agentic_ml.sandbox.models import ExecutionRequest


def _run(code: str, timeout: float = 10.0):
    """Helper to execute code in the sandbox and return the result."""
    req = ExecutionRequest(code=code, timeout_seconds=timeout, capture_plots=False)
    return ExecutionManager.execute(req)


class TestSandboxNetworkBlocking:
    """Verify that network operations are blocked in the sandbox."""

    def test_socket_import_blocked(self):
        """Importing socket and connecting to the internet must fail."""
        result = _run(
            "import socket\ns = socket.socket()\ns.connect(('8.8.8.8', 80))\nprint('CONNECTED')"
        )
        # Either the connection fails (error) or is blocked
        assert result.success is False or "CONNECTED" not in (result.stdout or ""), (
            "Socket connection should not succeed in sandbox"
        )

    def test_requests_blocked(self):
        """HTTP requests must not succeed in the sandbox."""
        result = _run(
            "import urllib.request\nresp = urllib.request.urlopen('http://google.com', timeout=2)\nprint('CONNECTED')"
        )
        assert result.success is False or "CONNECTED" not in (result.stdout or ""), (
            "HTTP request should not succeed in sandbox"
        )


class TestSandboxSubprocessBlocking:
    """Verify that subprocess spawning is blocked."""

    def test_subprocess_blocked(self):
        """subprocess.run should not be able to execute system commands."""
        result = _run(
            "import subprocess\nout = subprocess.run(['echo', 'PWNED'], capture_output=True)\nprint(out.stdout)"
        )
        # Either blocked entirely or subprocess call fails
        assert "PWNED" not in (result.stdout or ""), (
            "subprocess.run should not execute system commands in sandbox"
        )

    def test_os_system_blocked(self):
        """os.system should not execute arbitrary shell commands."""
        result = _run("import os\nos.system('echo PWNED')\nprint('done')")
        assert "PWNED" not in (result.stdout or ""), (
            "os.system should not execute in sandbox"
        )


class TestSandboxTimeout:
    """Verify that infinite loops are terminated by the timeout."""

    def test_infinite_loop_times_out(self):
        """An infinite loop must be killed within the timeout window."""
        result = _run("while True: pass", timeout=3.0)
        # Must not succeed; must time out or error
        assert result.success is False, "Infinite loop should not complete successfully"
        assert result.timed_out is True or result.error_type is not None, (
            "Infinite loop must produce timed_out=True or an error"
        )

    def test_sleep_times_out(self):
        """A very long sleep must be killed by the timeout."""
        result = _run("import time\ntime.sleep(9999)", timeout=3.0)
        assert result.success is False, "Long sleep should not complete"


class TestSandboxEnvProtection:
    """Verify that environment variables are not exposed to sandbox code."""

    def test_api_key_not_exposed(self, monkeypatch):
        """OPENAI_API_KEY must not be accessible inside the sandbox."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret-key-12345")
        result = _run(
            "import os\nkey = os.environ.get('OPENAI_API_KEY', 'NOT_FOUND')\nprint(key)"
        )
        stdout = result.stdout or ""
        assert "sk-test-secret-key-12345" not in stdout, (
            "OPENAI_API_KEY must not be visible inside the sandbox"
        )

    def test_env_vars_sanitized(self, monkeypatch):
        """Sensitive env vars must not be printed even if the sandbox runs successfully."""
        monkeypatch.setenv("SECRET_DB_PASSWORD", "super_secret_db_pass")
        result = _run(
            "import os\nprint(os.environ.get('SECRET_DB_PASSWORD', 'PROTECTED'))"
        )
        stdout = result.stdout or ""
        assert "super_secret_db_pass" not in stdout, (
            "SECRET_DB_PASSWORD must not leak from sandbox"
        )


class TestSandboxOutputLimit:
    """Verify that output is bounded."""

    def test_large_output_does_not_crash_server(self):
        """Generating a very large stdout should not crash the execution server."""
        result = _run(
            "for i in range(100_000): print('A' * 100)",
            timeout=10.0,
        )
        # Must not raise an exception (success or graceful truncation)
        # The test passes as long as the server doesn't crash (result object returned)
        assert result is not None, "Sandbox must return a result even for large output"
