"""
Deployment Agent Node.

Packages the winning model and fitted preprocessor into a secure, self-describing,
asymmetrically signed (Ed25519) production artifact bundle (artifacts/<model>/v<N>/).

Key responsibilities:
  1. Verify explicit approval invariant (AUTO_APPROVE or HUMAN_APPROVED)
  2. Call ArtifactBundleManager with full SLSA provenance context
  3. Verify the bundle immediately after creation (integrity + authenticity + model load check)
  4. Set deployment_completed=True ONLY after verification passes
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.types import Command
from langgraph.graph import END

from src.agentic_ml.state.agent_state import AgentState
from src.agentic_ml.llm.factory import get_llm
from src.agentic_ml.security.manifest import ArtifactBundleManager
from src.agentic_ml.core.context import RunContext

logger = logging.getLogger("agentic_ml.agents.deployment")

SYSTEM_PROMPT = (
    "You are the Deployment Agent. "
    "Package and verify the production artifact bundle with Ed25519 signing and "
    "SHA-256 manifest integrity. Confirm artifact size, version, and readiness for serving. "
    "Report integrity and authenticity status clearly."
)


def deployment_node(state: AgentState) -> Command:
    llm = get_llm()

    # Enforce approval invariant: deployment cannot proceed without explicit approval
    decision = state.get("deployment_decision")
    if decision not in {"AUTO_APPROVE", "HUMAN_APPROVED"}:
        raise RuntimeError(f"Deployment attempted without valid approval; got {decision!r}.")

    best_name = state.get("best_model_name")
    if not best_name:
        raise RuntimeError("Deployment requires a validated best_model_name from validation stage.")
    trained_models = state.get("trained_models") or {}
    best_model = trained_models.get(best_name)

    if best_model is None:
        raise RuntimeError(
            f"Deployment node: best_model '{best_name}' not found in trained_models — "
            "deployment_completed NOT set."
        )

    run_id = state.get("run_id")
    dataset_hash = state.get("dataset_hash")
    if not run_id or run_id == "unknown":
        raise RuntimeError("Deployment failed: missing or invalid run_id in state.")
    if not dataset_hash or dataset_hash == "unknown":
        raise RuntimeError("Deployment failed: missing or invalid dataset_hash in state.")

    # Package full pipeline (fitted preprocessor + model) for self-contained inference
    preprocessor = state.get("preprocessor_obj")
    if preprocessor is not None and hasattr(preprocessor, "transform"):
        from sklearn.pipeline import Pipeline
        deployable_obj = Pipeline([("preprocessor", preprocessor), ("model", best_model)])
    else:
        deployable_obj = best_model

    run_context = RunContext.create(
        run_id=run_id,
        dataset_hash=dataset_hash,
        random_seed=state.get("random_seed", 42),
        started_at=state.get("started_at"),
    )

    # ── Create SLSA-aligned signed artifact bundle ─────────────────────────
    bundle_info = ArtifactBundleManager.create_bundle(
        model_name=best_name,
        model_obj=deployable_obj,
        task_type=state.get("task_type", "classification"),
        feature_columns=state.get("selected_features"),
        target_column=state.get("target_column", "target"),
        metrics=state.get("best_model_metrics", {}),
        provenance=state.get("provenance", []),
        description=f"Automated build of {best_name} for task {state.get('task_type')}",
        run_context=run_context,
    )

    artifact_path = bundle_info["bundle_dir"]
    model_sha256 = bundle_info["hashes"].get("model.pkl", "")

    # ── Immediate post-creation verification ───────────────────────────────
    verification = ArtifactBundleManager.verify_bundle(artifact_path)
    if not verification["valid"]:
        raise RuntimeError(
            f"Artifact verification FAILED immediately after creation at {artifact_path}. "
            f"Errors: {verification['errors']} — deployment_completed NOT set."
        )

    logger.info(
        "Deployment: bundle verified ✓ -> %s [v=%s, sha256=%s..., onnx=%s]",
        artifact_path, bundle_info["version"], model_sha256[:12],
        bundle_info.get("onnx_exported", False),
    )

    execution_mode = state.get("execution_mode", "simulation")
    try:
        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Production artifact bundle created: {artifact_path}\n"
                    f"Version: {bundle_info['version']}\n"
                    f"SHA-256 (model.pkl): {model_sha256[:16]}...\n"
                    f"ONNX export: {bundle_info.get('onnx_exported', False)}\n"
                    f"Integrity: {verification['integrity_ok']}\n"
                    f"Authenticity (Ed25519): {verification['signature_ok']}\n"
                    f"Signed with: Ed25519"
                )
            ),
        ])
        execution_mode = "live"
    except Exception as exc:
        response = AIMessage(
            content=(
                f"[Deployment — Simulation Mode]\n"
                f"Artifact Bundle: {artifact_path}\n"
                f"Version: {bundle_info['version']}\n"
                f"SHA-256: {model_sha256[:16]}...\n"
                f"Integrity: ✓  Authenticity: ✓  (Ed25519)\n"
                f"LLM unavailable: {exc}"
            )
        )
        logger.warning("Deployment: simulation fallback — %s", exc)

    provenance_entry = {
        "agent_name": "deployment",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "operation": "Ed25519-signed artifact bundle generation + immediate verification",
        "result_summary": (
            f"model={best_name}, version={bundle_info['version']}, "
            f"sha256={model_sha256[:16]}, integrity={verification['integrity_ok']}, "
            f"authenticity={verification['signature_ok']}, "
            f"onnx={bundle_info.get('onnx_exported', False)}"
        ),
        "artifact_path": artifact_path,
    }

    evidence_entry = {
        "agent_name": "deployment",
        "decision": "DEPLOYED",
        "selected_tool": "ArtifactBundleManager.create_bundle + verify_bundle",
        "reason": (
            f"Bundle integrity verified (SHA-256 all files matched). "
            f"Ed25519 signature verified. Risk: {state.get('risk_level', 'LOW')} "
            f"({state.get('risk_score', 0)}/100)."
        ),
        "confidence": 1.0,
        "artifacts": [artifact_path],
        "metrics": bundle_info["hashes"],
        "warnings": [],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return Command(
        goto=END,
        update={
            "messages": [response],
            "artifact_path": artifact_path,
            "deployment_completed": True,
            "execution_mode": execution_mode,
            "provenance": [provenance_entry],
            "evidence": [evidence_entry],
        },
    )
