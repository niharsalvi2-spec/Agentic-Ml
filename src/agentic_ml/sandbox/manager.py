"""
ExecutionManager: coordinating facade for sandboxed code execution.
"""
import ast
import logging
from typing import Optional

from src.agentic_ml.sandbox.models import ExecutionRequest, ExecutionResult
from src.agentic_ml.sandbox.runner import SandboxRunner

logger = logging.getLogger("agentic_ml.sandbox.manager")


class ExecutionManager:
    """Coordinates code validation, sandbox runner dispatch, and result normalization."""

    @classmethod
    def execute(cls, request: ExecutionRequest) -> ExecutionResult:
        """
        Validate and execute code within the isolated sandbox environment.
        """
        code = request.code.strip()
        if not code:
            return ExecutionResult(
                success=True,
                stdout="",
                stderr="",
                returncode=0,
                execution_time_ms=0.0,
                images=[],
                timed_out=False,
            )

        # Pre-validate Python syntax to catch syntax errors before spinning up process
        try:
            ast.parse(code)
        except SyntaxError as syn_err:
            logger.info("Sandbox syntax error detected: %s", syn_err)
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=f"SyntaxError: {syn_err.msg} (line {syn_err.lineno})",
                returncode=1,
                execution_time_ms=0.0,
                error_type="syntax_error",
                timed_out=False,
            )

        return SandboxRunner.run(request)
