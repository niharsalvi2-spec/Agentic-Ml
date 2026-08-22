"""
Pipeline API Route — Thin proxy to RunManager.

This module's ONLY responsibility is HTTP boundary:
  - Validate request parameters
  - Delegate entirely to RunManager
  - Forward the SSE stream to the client

Zero hardcoded metrics. Zero synthetic code generators.
Zero knowledge of ML stages — that belongs to RunManager and LangGraph.
"""
import logging
from typing import Optional

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.agentic_ml.api.run_manager import stream_run, resume_run, get_run_status
from src.agentic_ml.sandbox.manager import ExecutionManager
from src.agentic_ml.sandbox.models import ExecutionRequest

# Compatibility export
generate_pipeline_events = stream_run

logger = logging.getLogger("agentic_ml.api.pipeline")
router = APIRouter()

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


# ── Request Models ─────────────────────────────────────────────────────────────

class PipelineRunRequest(BaseModel):
    prompt: str
    dataset_path: str = ""
    target_column: Optional[str] = None
    random_seed: int = 42


class ApprovalRequest(BaseModel):
    approved: bool


class ExecuteCodeRequest(BaseModel):
    code: str
    cell_id: str = "cell_1"


# ── SSE Streaming Endpoints ────────────────────────────────────────────────────

@router.post("/stream")
async def stream_pipeline_post(req: PipelineRunRequest):
    """
    Start a new pipeline run and stream live AgentEvents via SSE.

    The stream emits typed AgentEvent JSON objects (see core/events.py).
    The stream terminates with 'data: [DONE]'.
    """
    return StreamingResponse(
        stream_run(
            prompt=req.prompt,
            dataset_path=req.dataset_path,
            target_column=req.target_column,
            random_seed=req.random_seed,
        ),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


@router.get("/stream")
async def stream_pipeline_get(
    prompt: str = "Predict Customer Churn",
    dataset_path: str = "",
    target_column: Optional[str] = None,
    random_seed: int = 42,
):
    """GET variant for browser EventSource compatibility."""
    return StreamingResponse(
        stream_run(
            prompt=prompt,
            dataset_path=dataset_path,
            target_column=target_column,
            random_seed=random_seed,
        ),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


# ── HITL Resume Endpoint ───────────────────────────────────────────────────────

@router.post("/run/{run_id}/approve")
async def approve_deployment(run_id: str, req: ApprovalRequest):
    """
    Resume an interrupted (HITL) run after human approval or rejection.

    The response is another SSE stream that continues from the checkpoint
    and terminates with 'data: [DONE]'.
    """
    return StreamingResponse(
        resume_run(run_id=run_id, approved=req.approved),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


@router.get("/run/{run_id}/status")
async def get_pipeline_status(run_id: str):
    """Return the current status of a run."""
    return get_run_status(run_id)


# ── Code Execution Endpoint ────────────────────────────────────────────────────

@router.post("/execute-code")
async def execute_code_endpoint(req: ExecuteCodeRequest):
    """
    Live Python sandbox execution service via isolated ExecutionManager.
    Protects environment secrets, bounds memory and execution time,
    captures plots, and enforces workspace cleanup.
    """
    exec_req = ExecutionRequest(
        code=req.code,
        timeout_seconds=20.0,
        capture_plots=True,
    )
    result = ExecutionManager.execute(exec_req)
    status = "success" if result.success else "error"
    stdout_display = result.stdout or ("[✓ Executed — no stdout]" if status == "success" else "")

    return {
        "status": status,
        "stdout": stdout_display,
        "stderr": result.stderr,
        "images": result.images,
        "execution_time_ms": result.execution_time_ms,
        "cell_id": req.cell_id,
        "error_type": result.error_type,
        "timed_out": result.timed_out,
    }
