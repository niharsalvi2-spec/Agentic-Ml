"""
Unit tests for the Sandbox Execution Subsystem.
Verifies isolation, secret sanitization, plot capture, timeout enforcement, and error handling.
"""
import os
import unittest
from unittest.mock import patch

from src.agentic_ml.sandbox.models import ExecutionRequest, ExecutionResult
from src.agentic_ml.sandbox.runner import SandboxRunner
from src.agentic_ml.sandbox.manager import ExecutionManager


class TestSandboxExecution(unittest.TestCase):

    def test_successful_code_execution(self):
        req = ExecutionRequest(code="print('Hello Sandboxed World!')", timeout_seconds=5.0)
        result = ExecutionManager.execute(req)
        self.assertTrue(result.success)
        self.assertIn("Hello Sandboxed World!", result.stdout)
        self.assertEqual(result.returncode, 0)
        self.assertFalse(result.timed_out)

    def test_empty_code_execution(self):
        req = ExecutionRequest(code="   ")
        result = ExecutionManager.execute(req)
        self.assertTrue(result.success)
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr, "")

    def test_syntax_error_pre_validation(self):
        req = ExecutionRequest(code="def invalid_syntax(:")
        result = ExecutionManager.execute(req)
        self.assertFalse(result.success)
        self.assertEqual(result.error_type, "syntax_error")
        self.assertIn("SyntaxError", result.stderr)

    def test_runtime_exception_capture(self):
        req = ExecutionRequest(code="x = 1 / 0")
        result = ExecutionManager.execute(req)
        self.assertFalse(result.success)
        self.assertIn("ZeroDivisionError", result.stderr or result.stdout)

    def test_plot_capture(self):
        code = """
import matplotlib.pyplot as plt
plt.figure()
plt.plot([1, 2, 3], [4, 5, 6])
plt.title("Sandbox Test Plot")
"""
        req = ExecutionRequest(code=code, capture_plots=True, timeout_seconds=10.0)
        result = ExecutionManager.execute(req)
        self.assertTrue(result.success)
        self.assertGreaterEqual(len(result.images), 1)
        self.assertTrue(result.images[0].startswith("data:image/png;base64,"))

    def test_timeout_enforcement(self):
        code = """
import time
time.sleep(5)
print('Done sleeping')
"""
        req = ExecutionRequest(code=code, timeout_seconds=0.5)
        result = ExecutionManager.execute(req)
        self.assertFalse(result.success)
        self.assertTrue(result.timed_out)
        self.assertEqual(result.error_type, "timeout")

    def test_secret_isolation(self):
        # Set fake secrets in current environment
        with patch.dict(os.environ, {
            "GROQ_API_KEY": "super_secret_groq_key",
            "GEMINI_API_KEY": "super_secret_gemini_key",
            "SIGNING_PRIVATE_KEY": "private_key_material",
            "DATABASE_PASSWORD": "secret_db_pass",
        }):
            code = """
import os
groq = os.environ.get("GROQ_API_KEY", "NOT_FOUND")
gemini = os.environ.get("GEMINI_API_KEY", "NOT_FOUND")
signing = os.environ.get("SIGNING_PRIVATE_KEY", "NOT_FOUND")
db = os.environ.get("DATABASE_PASSWORD", "NOT_FOUND")
print(f"GROQ={groq};GEMINI={gemini};SIGNING={signing};DB={db}")
"""
            req = ExecutionRequest(code=code, timeout_seconds=5.0)
            result = ExecutionManager.execute(req)
            self.assertTrue(result.success)
            self.assertIn("GROQ=NOT_FOUND;GEMINI=NOT_FOUND;SIGNING=NOT_FOUND;DB=NOT_FOUND", result.stdout)

    def test_output_truncation_limit(self):
        code = "print('A' * 5000)"
        req = ExecutionRequest(code=code, max_output_bytes=1000, timeout_seconds=5.0)
        result = ExecutionManager.execute(req)
        self.assertTrue(result.success)
        self.assertLessEqual(len(result.stdout), 1100)
        self.assertIn("Output Truncated", result.stdout)


if __name__ == "__main__":
    unittest.main()
