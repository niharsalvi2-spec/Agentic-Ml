"""
Sandbox execution models for secure and isolated Python code execution.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ExecutionRequest(BaseModel):
    """Specification for isolated sandboxed execution."""
    code: str
    timeout_seconds: float = Field(default=15.0, ge=0.1, le=60.0)
    memory_limit_mb: int = Field(default=512, ge=64, le=2048)
    max_output_bytes: int = Field(default=100_000, ge=1_000, le=1_000_000)
    capture_plots: bool = True
    custom_env: Optional[Dict[str, str]] = None



class ExecutionResult(BaseModel):
    """Structured result returned by the Sandbox Execution Engine."""
    success: bool
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    execution_time_ms: float = 0.0
    images: List[str] = Field(default_factory=list)
    error_type: Optional[str] = None  # "timeout", "syntax_error", "runtime_error", etc.
    timed_out: bool = False
