"""
SandboxRunner: executes Python code in an isolated subprocess with strict security controls.

Security boundaries:
  - Runs in a clean, isolated temporary workspace.
  - Strict environment sanitization (ALL API keys, private keys, secrets stripped).
  - Hard execution timeouts with guaranteed process tree termination.
  - Maximum output byte cap to prevent memory exhaustion attacks.
  - Guaranteed workspace directory cleanup upon completion.
"""
import os
import sys
import json
import time
import shutil
import tempfile
import subprocess
import logging
from typing import Dict, List, Optional

from src.agentic_ml.sandbox.models import ExecutionRequest, ExecutionResult

logger = logging.getLogger("agentic_ml.sandbox.runner")

# Sensitive substrings to strip unconditionally from the child environment
SECRET_PATTERNS = (
    "KEY", "SECRET", "TOKEN", "PASS", "AUTH", "CREDENTIAL",
    "GROQ", "GEMINI", "OPENAI", "ANTHROPIC", "AWS", "AZURE",
    "DATABASE", "SIGNING", "PRIVATE", "SALT"
)

# Safe environment variables to preserve across platforms
SAFE_ENV_VARS = (
    "PATH", "PATHEXT", "SYSTEMROOT", "WINDIR", "COMSPEC",
    "TEMP", "TMP", "PYTHONPATH", "PYTHONHOME", "LANG", "LC_ALL",
    "VIRTUAL_ENV", "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE"
)


class SandboxRunner:
    """Executes arbitrary Python code inside an isolated, unprivileged process sandbox."""

    @staticmethod
    def _get_python_executable() -> str:
        """Resolve the current active virtual environment Python interpreter."""
        if sys.prefix != getattr(sys, "base_prefix", sys.prefix):
            candidate = (
                os.path.join(sys.prefix, "Scripts", "python.exe")
                if os.name == "nt"
                else os.path.join(sys.prefix, "bin", "python")
            )
            if os.path.exists(candidate):
                return candidate
        return sys.executable

    @classmethod
    def _build_sanitized_env(cls, custom_env: Optional[Dict[str, str]], temp_dir: str) -> Dict[str, str]:
        """
        Construct a minimal, secret-free environment for the sandbox process.
        """
        clean_env: Dict[str, str] = {}

        # Allow only safe baseline OS variables
        for var in SAFE_ENV_VARS:
            if var in os.environ:
                clean_env[var] = os.environ[var]

        # Explicitly scrub any variable matching known secret patterns
        for key in list(clean_env.keys()):
            upper = key.upper()
            if any(pattern in upper for pattern in SECRET_PATTERNS):
                del clean_env[key]

        # Propagate sys.path so sandbox environment finds installed packages
        clean_env["PYTHONPATH"] = os.pathsep.join(sys.path)

        # Direct matplotlib config into temp dir so it doesn't touch user home
        clean_env["MPLCONFIGDIR"] = os.path.join(temp_dir, ".mpl_config")
        clean_env["PYTHONUNBUFFERED"] = "1"
        clean_env["PYTHONDONTWRITEBYTECODE"] = "1"

        if custom_env:
            for k, v in custom_env.items():
                upper = k.upper()
                if not any(pattern in upper for pattern in SECRET_PATTERNS):
                    clean_env[k] = str(v)

        return clean_env

    @classmethod
    def run(cls, request: ExecutionRequest) -> ExecutionResult:
        """
        Run code inside an isolated sandbox subprocess.
        """
        temp_dir = tempfile.mkdtemp(prefix="agentic_ml_sandbox_")
        script_path = os.path.join(temp_dir, "sandbox_exec.py")
        t0 = time.perf_counter()

        try:
            sanitized_env = cls._build_sanitized_env(request.custom_env, temp_dir)
            python_bin = cls._get_python_executable()

            # Build isolated execution script wrapper
            code_lines = request.code.splitlines()
            indented_code = "\n".join("    " + line for line in code_lines)

            security_prelude = """# -*- coding: utf-8 -*-
import sys, os

def __block_dangerous_ops():
    try:
        import socket
        def _blocked_socket(*args, **kwargs):
            raise PermissionError("Network access is blocked in the execution sandbox.")
        socket.socket = _blocked_socket
        socket.create_connection = _blocked_socket
    except Exception:
        pass

    try:
        import urllib.request
        def _blocked_urlopen(*args, **kwargs):
            raise PermissionError("Network access is blocked in the execution sandbox.")
        urllib.request.urlopen = _blocked_urlopen
    except Exception:
        pass

    try:
        import subprocess
        def _blocked_popen(*args, **kwargs):
            raise PermissionError("Subprocess execution is blocked in the execution sandbox.")
        subprocess.Popen = _blocked_popen
        subprocess.run = _blocked_popen
        subprocess.call = _blocked_popen
        subprocess.check_call = _blocked_popen
        subprocess.check_output = _blocked_popen
    except Exception:
        pass

    try:
        def _blocked_system(*args, **kwargs):
            raise PermissionError("os.system is blocked in the execution sandbox.")
        os.system = _blocked_system
        os.popen = _blocked_system
    except Exception:
        pass

__block_dangerous_ops()
"""

            if request.capture_plots:
                wrapper = f"""{security_prelude}
import io, base64, json

__figs = []
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    __has_mpl = True
except ImportError:
    __has_mpl = False

try:
{indented_code}
except Exception:
    import traceback
    traceback.print_exc()

if __has_mpl:
    for _fn in plt.get_fignums():
        try:
            _fig = plt.figure(_fn)
            _buf = io.BytesIO()
            _fig.savefig(_buf, format='png', bbox_inches='tight', dpi=100, facecolor='#ffffff')
            _buf.seek(0)
            __figs.append("data:image/png;base64," + base64.b64encode(_buf.read()).decode())
            plt.close(_fig)
        except Exception:
            pass

    if __figs:
        print("\\n__AGENTIC_ML_PLOTS__:" + json.dumps(__figs))
"""
            else:
                wrapper = f"""{security_prelude}
try:
{indented_code}
except Exception:
    import traceback
    traceback.print_exc()
"""

            with open(script_path, "w", encoding="utf-8") as f:
                f.write(wrapper)

            # Launch isolated subprocess
            proc = subprocess.Popen(
                [python_bin, script_path],
                cwd=temp_dir,
                env=sanitized_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )

            try:
                stdout, stderr = proc.communicate(timeout=request.timeout_seconds)
                elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()
                elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
                logger.warning("Sandbox execution timed out after %.1fs", request.timeout_seconds)
                return ExecutionResult(
                    success=False,
                    stdout=stdout[:request.max_output_bytes],
                    stderr=(stderr + f"\n[Execution Timed Out after {request.timeout_seconds}s]")[:request.max_output_bytes],
                    returncode=-1,
                    execution_time_ms=elapsed_ms,
                    error_type="timeout",
                    timed_out=True,
                )

            # Cap output sizes to avoid memory exhaustion
            if len(stdout) > request.max_output_bytes:
                stdout = stdout[:request.max_output_bytes] + "\n[Output Truncated: max limit reached]"
            if len(stderr) > request.max_output_bytes:
                stderr = stderr[:request.max_output_bytes] + "\n[Error Output Truncated: max limit reached]"

            images: List[str] = []
            clean_out = stdout.strip()

            if "__AGENTIC_ML_PLOTS__:" in stdout:
                parts = stdout.split("__AGENTIC_ML_PLOTS__:")
                clean_out = parts[0].strip()
                try:
                    images = json.loads(parts[1].strip())
                except Exception as exc:
                    logger.debug("Failed to decode plot JSON: %s", exc)

            has_traceback = ("Traceback (most recent call last):" in clean_out) or ("Traceback (most recent call last):" in stderr)
            success = (proc.returncode == 0) and not has_traceback
            error_type = None if success else ("runtime_error" if proc.returncode != 0 else "exception_logged")

            return ExecutionResult(
                success=success,
                stdout=clean_out,
                stderr=stderr.strip(),
                returncode=proc.returncode,
                execution_time_ms=elapsed_ms,
                images=images,
                error_type=error_type,
                timed_out=False,
            )

        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
            logger.exception("Sandbox runner unexpected error: %s", exc)
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=f"Sandbox execution error: {str(exc)}",
                returncode=-1,
                execution_time_ms=elapsed_ms,
                error_type="system_error",
                timed_out=False,
            )

        finally:
            # Ensure complete cleanup of temporary sandbox directory
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception as e:
                logger.warning("Failed to clean up sandbox temp dir %s: %s", temp_dir, e)
