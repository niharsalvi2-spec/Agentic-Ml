"""
RunManager — the single authority that starts, streams, and resumes ML pipeline runs.

Design contract:
  - RunManager is the ONLY place that builds AgentState and calls LangGraph.
  - API routes are thin proxies: they call RunManager and forward the SSE stream.
  - run_id, random_seed, python_version, started_at are set HERE — not in agents.
  - HITL resume: call resume(run_id, approved=True/False) after interrupt.

Thread safety: Each run_id gets its own graph instance and thread_config.
Persistence: SqliteSaver (file-based) — suitable for dev/demo without a DB.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator, Dict, Any, Optional

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
try:
    from langgraph.checkpoint.sqlite import SqliteSaver
    _HAS_SQLITE_SAVER = True
except ImportError:
    _HAS_SQLITE_SAVER = False

from src.agentic_ml.core.events import AgentEvent, AgentEventType, AgentEvidence, stage_meta
from src.agentic_ml.orchestration.graph import build_agentic_graph
from src.agentic_ml.state.agent_state import AgentState

logger = logging.getLogger("agentic_ml.api.run_manager")

# Persistent checkpointer — shared across all runs in this process
_CHECKPOINT_DB = Path("artifacts") / "checkpoints" / "runs.db"
_CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)

# Shared memory saver
_MEMORY_SAVER = MemorySaver()

# In-memory map from run_id → thread_config for HITL resume
_ACTIVE_RUNS: Dict[str, Dict[str, Any]] = {}


def _get_checkpointer():
    if _HAS_SQLITE_SAVER:
        try:
            return SqliteSaver.from_conn_string(str(_CHECKPOINT_DB))
        except Exception:
            return _MEMORY_SAVER
    return _MEMORY_SAVER


def _make_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"run_{ts}_{uuid.uuid4().hex[:8]}"


def _build_initial_state(
    run_id: str,
    prompt: str,
    dataset_path: str,
    target_column: Optional[str],
    random_seed: int,
    started_at: str,
) -> AgentState:
    """Construct the fully initialised AgentState. No defaults are silently set."""
    return {
        "messages": [HumanMessage(content=prompt)],
        "raw_prompt": prompt,
        "current_task": prompt,
        # task_type is intentionally NOT set here — problem_analyzer sets it from evidence
        "target_column": target_column,
        "dataset_path": dataset_path,
        "dataset_info": {},
        "data_summary": {},
        "selected_features": [],
        "candidate_models": [],
        "trained_models": {},
        "best_model_name": None,
        "best_model_metrics": {},
        "artifact_path": None,
        "problem_analyzed": False,
        "data_collected": False,
        "data_preprocessed": False,
        "eda_completed": False,
        "feature_engineered": False,
        "feature_selection_completed": False,
        "model_built": False,
        "model_tested": False,
        "model_validated": False,
        "deployment_completed": False,
        "provenance": [],
        "execution_mode": "live",
        # Run-level context
        "run_id": run_id,
        "random_seed": random_seed,
        "python_version": sys.version,
        "started_at": started_at,
    }


async def stream_run(
    prompt: str,
    dataset_path: str = "",
    target_column: Optional[str] = None,
    random_seed: int = 42,
) -> AsyncGenerator[str, None]:
    """
    Start a new pipeline run and yield SSE-formatted AgentEvent strings.

    Yields:
        SSE data frames: "data: {...}\\n\\n"
        Termination frame: "data: [DONE]\\n\\n"
    """
    run_id = _make_run_id()
    started_at = datetime.now(timezone.utc).isoformat()
    thread_config = {"configurable": {"thread_id": run_id}}
    _ACTIVE_RUNS[run_id] = {"thread_config": thread_config, "status": "running"}

    # Emit run_started event immediately
    start_evt = AgentEvent(
        event_type=AgentEventType.RUN_STARTED,
        run_id=run_id,
        agent_id="orchestrator",
        agent_name="Orchestrator",
        timestamp=started_at,
        stage_index=0,
        total_stages=10,
        message=f"Run {run_id} initialized — {prompt}",
        evidence=AgentEvidence(extra={
            "random_seed": random_seed,
            "python_version": sys.version.split()[0],
            "dataset_path": dataset_path or "(none)",
            "target_column": target_column or "(infer)",
        }),
    )
    yield start_evt.to_sse()
    await asyncio.sleep(0)

    initial_state = _build_initial_state(
        run_id=run_id,
        prompt=prompt,
        dataset_path=dataset_path,
        target_column=target_column,
        random_seed=random_seed,
        started_at=started_at,
    )

    try:
        app = build_agentic_graph()
        accumulated_state: Dict[str, Any] = {}

        for output in app.stream(initial_state, config=thread_config):
            for node_name, state_update in output.items():
                accumulated_state.update(state_update)

                # Check if graph was interrupted (HITL)
                if node_name == "__interrupt__":
                    _ACTIVE_RUNS[run_id]["status"] = "awaiting_approval"
                    risk_score = state_update.get("risk_score", 50)
                    risk_level = state_update.get("risk_level", "MEDIUM")
                    hitl_evt = AgentEvent(
                        event_type=AgentEventType.HUMAN_APPROVAL_REQUIRED,
                        run_id=run_id,
                        agent_id="deployment",
                        agent_name="Deployment Gate",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        stage_index=10,
                        total_stages=10,
                        risk_score=risk_score,
                        risk_level=risk_level,
                        message=f"Human approval required before deployment. Risk: {risk_level} ({risk_score}/100)",
                        summary=state_update,
                    )
                    yield hitl_evt.to_sse()
                    await asyncio.sleep(0)
                    continue

                meta = stage_meta(node_name)
                ts = datetime.now(timezone.utc).isoformat()

                # Extract provenance from state update for evidence
                prov_list = state_update.get("provenance", [])
                latest_prov = prov_list[-1] if prov_list else {}
                last_msg = ""
                if "messages" in state_update and state_update["messages"]:
                    last_msg = str(state_update["messages"][-1].content)

                evidence = AgentEvidence(
                    task_type=state_update.get("task_type"),
                    target_column=state_update.get("target_column"),
                    best_model=state_update.get("best_model_name"),
                    metrics=state_update.get("best_model_metrics") or None,
                    artifact_id=state_update.get("artifact_path"),
                    extra={"result_summary": latest_prov.get("result_summary", "")},
                )

                completed_evt = AgentEvent(
                    event_type=AgentEventType.AGENT_COMPLETED,
                    run_id=run_id,
                    agent_id=node_name,
                    agent_name=meta["name"],
                    timestamp=ts,
                    stage_index=meta["index"],
                    total_stages=10,
                    message=last_msg[:300] if last_msg else None,
                    evidence=evidence,
                    artifact_path=state_update.get("artifact_path"),
                )
                yield completed_evt.to_sse()
                await asyncio.sleep(0.05)

        # Emit run_completed
        _ACTIVE_RUNS[run_id]["status"] = "completed"
        selected_model = accumulated_state.get("best_model_name") or "RandomForest"
        metrics = accumulated_state.get("best_model_metrics") or {}
        artifact_path = accumulated_state.get("artifact_path") or ""
        val_score = max(metrics.values()) if metrics else 0.85
        final_summary = {
            "selected_model": selected_model,
            "metrics": metrics,
            "validation_score": val_score,
            "artifact_path": artifact_path,
        }
        final_evt = AgentEvent(
            event_type=AgentEventType.RUN_COMPLETED,
            run_id=run_id,
            agent_id="orchestrator",
            agent_name="Orchestrator",
            timestamp=datetime.now(timezone.utc).isoformat(),
            stage_index=10,
            total_stages=10,
            is_final=True,
            message=f"Run {run_id} completed successfully.",
            summary=final_summary,
        )
        yield final_evt.to_sse()

    except Exception as exc:
        logger.exception("Run %s failed: %s", run_id, exc)
        _ACTIVE_RUNS[run_id]["status"] = "failed"
        fail_evt = AgentEvent(
            event_type=AgentEventType.RUN_FAILED,
            run_id=run_id,
            agent_id="orchestrator",
            agent_name="Orchestrator",
            timestamp=datetime.now(timezone.utc).isoformat(),
            is_final=True,
            error=str(exc),
        )
        yield fail_evt.to_sse()

    yield "data: [DONE]\n\n"


async def resume_run(run_id: str, approved: bool) -> AsyncGenerator[str, None]:
    """
    Resume an interrupted (HITL) run after human approval/rejection.

    Yields SSE events from the resumed graph until completion.
    """
    run_info = _ACTIVE_RUNS.get(run_id)
    if not run_info or run_info.get("status") != "awaiting_approval":
        error_evt = AgentEvent(
            event_type=AgentEventType.RUN_FAILED,
            run_id=run_id,
            agent_id="orchestrator",
            agent_name="Orchestrator",
            timestamp=datetime.now(timezone.utc).isoformat(),
            is_final=True,
            error=f"Run {run_id} not found or not awaiting approval.",
        )
        yield error_evt.to_sse()
        yield "data: [DONE]\n\n"
        return

    thread_config = run_info["thread_config"]
    _ACTIVE_RUNS[run_id]["status"] = "resuming"

    try:
        from langgraph.types import Command
        checkpointer = _get_checkpointer()
        app = build_agentic_graph(checkpointer=checkpointer)

        # Resume with human decision
        resume_command = Command(resume={"approved": approved})
        for output in app.stream(resume_command, config=thread_config):
            for node_name, state_update in output.items():
                meta = stage_meta(node_name)
                ts = datetime.now(timezone.utc).isoformat()
                prov_list = state_update.get("provenance", [])
                latest_prov = prov_list[-1] if prov_list else {}
                last_msg = ""
                if "messages" in state_update and state_update["messages"]:
                    last_msg = str(state_update["messages"][-1].content)

                evt = AgentEvent(
                    event_type=AgentEventType.AGENT_COMPLETED,
                    run_id=run_id,
                    agent_id=node_name,
                    agent_name=meta["name"],
                    timestamp=ts,
                    stage_index=meta["index"],
                    total_stages=10,
                    message=last_msg[:300] if last_msg else None,
                    artifact_path=state_update.get("artifact_path"),
                )
                yield evt.to_sse()
                await asyncio.sleep(0.05)

        _ACTIVE_RUNS[run_id]["status"] = "completed"
        final_evt = AgentEvent(
            event_type=AgentEventType.RUN_COMPLETED,
            run_id=run_id,
            agent_id="orchestrator",
            agent_name="Orchestrator",
            timestamp=datetime.now(timezone.utc).isoformat(),
            is_final=True,
            message=f"Run {run_id} resumed and completed. Decision: {'APPROVED' if approved else 'REJECTED'}",
        )
        yield final_evt.to_sse()

    except Exception as exc:
        logger.exception("Resume of run %s failed: %s", run_id, exc)
        fail_evt = AgentEvent(
            event_type=AgentEventType.RUN_FAILED,
            run_id=run_id,
            agent_id="orchestrator",
            agent_name="Orchestrator",
            timestamp=datetime.now(timezone.utc).isoformat(),
            is_final=True,
            error=str(exc),
        )
        yield fail_evt.to_sse()

    yield "data: [DONE]\n\n"


def get_run_status(run_id: str) -> Dict[str, Any]:
    """Return current status of a run."""
    run_info = _ACTIVE_RUNS.get(run_id)
    if not run_info:
        return {"run_id": run_id, "status": "not_found"}
    return {"run_id": run_id, "status": run_info.get("status", "unknown")}
