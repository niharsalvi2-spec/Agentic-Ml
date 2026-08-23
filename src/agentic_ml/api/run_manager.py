"""
RunManager — The single authority that starts, streams, and resumes ML pipeline runs.

Design contract:
  - RunManager is the ONLY place that builds AgentState and calls LangGraph.
  - API routes are thin proxies: they call RunManager and forward the SSE stream.
  - run_id, random_seed, python_version, started_at are set HERE — not in agents.
  - Checkpoint persistence is mandatory and configurable via CHECKPOINT_BACKEND ("sqlite" | "memory").
  - NO silent fallback from sqlite to memory.
  - Monotonic sequence numbers across initial stream and HITL resume.
  - Fail-closed: runs only succeed if verifiable evidence exists.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import pickle
import sqlite3
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator, Dict, Any, Optional

from langgraph.checkpoint.base import SerializerProtocol
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from src.agentic_ml.core.events import AgentEvent, AgentEventType, AgentEvidence, stage_meta
from src.agentic_ml.orchestration.graph import build_agentic_graph
from src.agentic_ml.orchestration.completion_gate import verify_run_completion
from src.agentic_ml.state.agent_state import AgentState
from src.agentic_ml.storage.run_registry import RunRegistry
from src.agentic_ml.ml_engine.evaluation.metrics import extract_primary_metric

logger = logging.getLogger("agentic_ml.api.run_manager")

_CHECKPOINT_CONN: Optional[sqlite3.Connection] = None
_DEV_MEMORY_SAVER: Optional[MemorySaver] = None


class MLCheckpointSerializer(SerializerProtocol):
    """
    Serializer that encodes ML objects (pandas DataFrames, numpy arrays, sklearn models)
    alongside jsonplus primitive structures.
    """
    def __init__(self) -> None:
        self._jsonplus = JsonPlusSerializer()

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        try:
            return self._jsonplus.dumps_typed(obj)
        except Exception:
            return ("pickle", pickle.dumps(obj))

    def loads_typed(self, type_and_data: tuple[str, bytes]) -> Any:
        type_, data = type_and_data
        if type_ == "pickle":
            return pickle.loads(data)
        return self._jsonplus.loads_typed(type_and_data)


def _get_checkpointer_config() -> tuple[str, Path]:
    backend = os.environ.get("CHECKPOINT_BACKEND", "sqlite").lower().strip()
    db_path_str = os.environ.get("CHECKPOINT_DB_PATH", "artifacts/checkpoints/runs.db")
    db_path = Path(db_path_str)
    return backend, db_path


def get_checkpointer():
    """
    Get configured checkpointer instance.
    Raises RuntimeError if persistent backend is requested but unavailable.
    """
    global _CHECKPOINT_CONN, _DEV_MEMORY_SAVER
    backend, db_path = _get_checkpointer_config()

    if backend == "memory":
        if _DEV_MEMORY_SAVER is None:
            _DEV_MEMORY_SAVER = MemorySaver()
        return _DEV_MEMORY_SAVER

    if backend == "sqlite":
        try:
            db_path.parent.mkdir(parents=True, exist_ok=True)
            if _CHECKPOINT_CONN is None:
                _CHECKPOINT_CONN = sqlite3.connect(str(db_path), check_same_thread=False, timeout=30.0)
                _CHECKPOINT_CONN.execute("PRAGMA journal_mode=WAL;")
                _CHECKPOINT_CONN.execute("PRAGMA synchronous=NORMAL;")
            return SqliteSaver(_CHECKPOINT_CONN, serde=MLCheckpointSerializer())
        except Exception as exc:
            logger.error("Failed to initialize SQLite checkpointer at %s: %s", db_path, exc)
            raise RuntimeError(
                f"SQLite checkpointer failed to initialize at '{db_path}'. "
                "Silent fallback to memory is forbidden in production/sqlite mode."
            ) from exc

    raise ValueError(
        f"Invalid CHECKPOINT_BACKEND '{backend}'. Must be 'sqlite' or 'memory'."
    )


def reset_checkpointer():
    """Reset checkpointer connection (for testing)."""
    global _CHECKPOINT_CONN, _DEV_MEMORY_SAVER
    if _CHECKPOINT_CONN is not None:
        try:
            _CHECKPOINT_CONN.close()
        except Exception:
            pass
        _CHECKPOINT_CONN = None
    _DEV_MEMORY_SAVER = None


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
    """Construct the fully initialized AgentState."""
    return {
        "messages": [HumanMessage(content=prompt)],
        "raw_prompt": prompt,
        "current_task": prompt,
        "target_column": target_column,
        "target_inference_method": "explicit" if target_column else "deferred",
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
        "validation_retry_count": 0,
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
    """
    run_id = _make_run_id()
    started_at = datetime.now(timezone.utc).isoformat()
    thread_config = {"configurable": {"thread_id": run_id}}

    # Persist in RunRegistry
    registry = RunRegistry.get()
    registry.create_run(run_id, prompt, dataset_path, target_column, random_seed)

    seq_num = 1

    def make_event(event_type: AgentEventType, **kwargs) -> AgentEvent:
        nonlocal seq_num
        evt = AgentEvent(
            event_type=event_type,
            event_id=f"evt_{run_id}_{seq_num:04d}",
            sequence_number=seq_num,
            run_id=run_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            **kwargs,
        )
        seq_num += 1
        return evt

    # Emit run_started event immediately
    start_evt = make_event(
        event_type=AgentEventType.RUN_STARTED,
        agent_id="orchestrator",
        agent_name="Orchestrator",
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
        checkpointer = get_checkpointer()
        app = build_agentic_graph(checkpointer=checkpointer)
        accumulated_state: Dict[str, Any] = {}

        for output in app.stream(initial_state, config=thread_config):
            for node_name, state_update in output.items():
                accumulated_state.update(state_update)

                # Check if graph was interrupted (HITL)
                if node_name == "__interrupt__":
                    risk_score = state_update.get("risk_score", 50)
                    risk_level = state_update.get("risk_level", "MEDIUM")
                    registry.record_hitl_request(run_id, risk_score, risk_level)
                    registry.update_status(run_id, "AWAITING_APPROVAL", last_sequence_number=seq_num)

                    hitl_evt = make_event(
                        event_type=AgentEventType.HUMAN_APPROVAL_REQUIRED,
                        agent_id="deployment_gate",
                        agent_name="Deployment Gate",
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

                prov_list = state_update.get("provenance", [])
                latest_prov = prov_list[-1] if prov_list else {}
                last_msg = ""
                if "messages" in state_update and state_update["messages"]:
                    last_msg = str(state_update["messages"][-1].content)

                evidence = AgentEvidence(
                    task_type=state_update.get("task_type"),
                    target_column=state_update.get("target_column"),
                    target_inference=state_update.get("target_inference_method"),
                    best_model=state_update.get("best_model_name"),
                    metrics=state_update.get("best_model_metrics") or None,
                    artifact_id=state_update.get("artifact_path"),
                    extra={"result_summary": latest_prov.get("result_summary", "")},
                )

                completed_evt = make_event(
                    event_type=AgentEventType.AGENT_COMPLETED,
                    agent_id=node_name,
                    agent_name=meta["name"],
                    stage_index=meta["index"],
                    total_stages=10,
                    message=last_msg[:300] if last_msg else None,
                    evidence=evidence,
                    artifact_path=state_update.get("artifact_path"),
                )
                yield completed_evt.to_sse()
                await asyncio.sleep(0.05)

        # Truthful completion verification
        is_completed, failures = verify_run_completion(accumulated_state)

        if is_completed:
            selected_model = accumulated_state["best_model_name"]
            metrics = accumulated_state["best_model_metrics"]
            artifact_path = accumulated_state["artifact_path"]

            task_type = accumulated_state.get("task_type", "classification")
            pm = extract_primary_metric(metrics, task_type=task_type)
            val_score = pm.value

            final_summary = {
                "selected_model": selected_model,
                "metrics": metrics,
                "primary_metric": pm.name,
                "validation_score": val_score,
                "artifact_path": artifact_path,
                "risk_score": accumulated_state.get("risk_score", 0),
                "risk_level": accumulated_state.get("risk_level", "LOW"),
            }
            registry.update_status(run_id, "COMPLETED", artifact_path=artifact_path, last_sequence_number=seq_num)

            final_evt = make_event(
                event_type=AgentEventType.RUN_COMPLETED,
                agent_id="orchestrator",
                agent_name="Orchestrator",
                stage_index=10,
                total_stages=10,
                is_final=True,
                message=f"Run {run_id} completed successfully — artifact generated at {artifact_path}.",
                summary=final_summary,
            )
            yield final_evt.to_sse()

        elif accumulated_state.get("deployment_decision") == "REJECTED":
            registry.update_status(run_id, "REJECTED", error="Deployment rejected by reviewer.", last_sequence_number=seq_num)
            rejected_evt = make_event(
                event_type=AgentEventType.RUN_FAILED,
                agent_id="deployment_gate",
                agent_name="Deployment Gate",
                stage_index=10,
                total_stages=10,
                is_final=True,
                message=f"Run {run_id} halted: Deployment rejected by reviewer.",
                error="Deployment rejected by reviewer.",
            )
            yield rejected_evt.to_sse()

        else:
            error_msg = "; ".join(failures) if failures else "Pipeline execution halted due to unfulfilled completion invariants."
            registry.update_status(run_id, "FAILED", error=error_msg, last_sequence_number=seq_num)
            fail_evt = make_event(
                event_type=AgentEventType.RUN_FAILED,
                agent_id="orchestrator",
                agent_name="Orchestrator",
                stage_index=10,
                total_stages=10,
                is_final=True,
                error=error_msg,
            )
            yield fail_evt.to_sse()

    except Exception as exc:
        logger.exception("Run %s failed: %s", run_id, exc)
        registry.update_status(run_id, "FAILED", error=str(exc), last_sequence_number=seq_num)
        fail_evt = make_event(
            event_type=AgentEventType.RUN_FAILED,
            agent_id="orchestrator",
            agent_name="Orchestrator",
            is_final=True,
            error=str(exc),
        )
        yield fail_evt.to_sse()

    yield "data: [DONE]\n\n"


async def resume_run(run_id: str, approved: bool) -> AsyncGenerator[str, None]:
    """
    Resume an interrupted (HITL) run after human approval/rejection.
    Enforces exactly-once compare-and-set and monotonic event sequencing.
    """
    registry = RunRegistry.get()
    run_record = registry.get_run(run_id)

    if not run_record:
        error_evt = AgentEvent(
            event_type=AgentEventType.RUN_FAILED,
            run_id=run_id,
            agent_id="orchestrator",
            agent_name="Orchestrator",
            timestamp=datetime.now(timezone.utc).isoformat(),
            is_final=True,
            error=f"Run {run_id} not found in persistent registry.",
        )
        yield error_evt.to_sse()
        yield "data: [DONE]\n\n"
        return

    # Atomic compare-and-set transition
    success_transition = registry.resolve_hitl_approval(run_id, approved)
    if not success_transition:
        status = run_record.get("status")
        error_evt = AgentEvent(
            event_type=AgentEventType.RUN_FAILED,
            run_id=run_id,
            agent_id="orchestrator",
            agent_name="Orchestrator",
            timestamp=datetime.now(timezone.utc).isoformat(),
            is_final=True,
            error=f"Approval rejected: Run {run_id} already resolved or in status '{status}'.",
        )
        yield error_evt.to_sse()
        yield "data: [DONE]\n\n"
        return

    thread_config = {"configurable": {"thread_id": run_id}}
    seq_num = max(1, int(run_record.get("last_sequence_number") or 0)) + 1

    def make_event(event_type: AgentEventType, **kwargs) -> AgentEvent:
        nonlocal seq_num
        evt = AgentEvent(
            event_type=event_type,
            event_id=f"evt_{run_id}_{seq_num:04d}",
            sequence_number=seq_num,
            run_id=run_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            **kwargs,
        )
        seq_num += 1
        return evt

    try:
        from langgraph.types import Command
        checkpointer = get_checkpointer()
        app = build_agentic_graph(checkpointer=checkpointer)

        accumulated_state: Dict[str, Any] = {}
        resume_command = Command(resume={"approved": approved})

        for output in app.stream(resume_command, config=thread_config):
            for node_name, state_update in output.items():
                accumulated_state.update(state_update)
                meta = stage_meta(node_name)
                prov_list = state_update.get("provenance", [])
                latest_prov = prov_list[-1] if prov_list else {}
                last_msg = ""
                if "messages" in state_update and state_update["messages"]:
                    last_msg = str(state_update["messages"][-1].content)

                evt = make_event(
                    event_type=AgentEventType.AGENT_COMPLETED,
                    agent_id=node_name,
                    agent_name=meta["name"],
                    stage_index=meta["index"],
                    total_stages=10,
                    message=last_msg[:300] if last_msg else None,
                    artifact_path=state_update.get("artifact_path"),
                )
                yield evt.to_sse()
                await asyncio.sleep(0.05)

        is_completed, failures = verify_run_completion(accumulated_state)

        if is_completed:
            artifact_path = accumulated_state.get("artifact_path", "")
            registry.update_status(run_id, "COMPLETED", artifact_path=artifact_path, last_sequence_number=seq_num)
            final_evt = make_event(
                event_type=AgentEventType.RUN_COMPLETED,
                agent_id="orchestrator",
                agent_name="Orchestrator",
                stage_index=10,
                total_stages=10,
                is_final=True,
                message=f"Run {run_id} resumed and completed. Decision: {'APPROVED' if approved else 'REJECTED'}",
                summary={
                    "selected_model": accumulated_state.get("best_model_name"),
                    "metrics": accumulated_state.get("best_model_metrics", {}),
                    "artifact_path": artifact_path,
                },
            )
            yield final_evt.to_sse()
        else:
            error_msg = "; ".join(failures) if failures else "Deployment failed after resume."
            registry.update_status(run_id, "FAILED", error=error_msg, last_sequence_number=seq_num)
            fail_evt = make_event(
                event_type=AgentEventType.RUN_FAILED,
                agent_id="orchestrator",
                agent_name="Orchestrator",
                stage_index=10,
                total_stages=10,
                is_final=True,
                error=error_msg,
            )
            yield fail_evt.to_sse()

    except Exception as exc:
        logger.exception("Resume of run %s failed: %s", run_id, exc)
        registry.update_status(run_id, "FAILED", error=str(exc), last_sequence_number=seq_num)
        fail_evt = make_event(
            event_type=AgentEventType.RUN_FAILED,
            agent_id="orchestrator",
            agent_name="Orchestrator",
            is_final=True,
            error=str(exc),
        )
        yield fail_evt.to_sse()

    yield "data: [DONE]\n\n"


def get_run_status(run_id: str) -> Dict[str, Any]:
    """Return durable current status of a run from registry."""
    run_record = RunRegistry.get().get_run(run_id)
    if not run_record:
        return {"run_id": run_id, "status": "not_found"}
    return {
        "run_id": run_id,
        "status": run_record["status"],
        "risk_score": run_record["risk_score"],
        "risk_level": run_record["risk_level"],
        "deployment_decision": run_record["deployment_decision"],
        "artifact_path": run_record["artifact_path"],
        "error": run_record["error"],
    }
