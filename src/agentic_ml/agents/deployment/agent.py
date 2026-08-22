"""
Deployment Agent Node.

Packages the winning model into a secure, self-describing, SHA-256 integrity-
verified production artifact bundle. Sets deployment_completed=True only after
PKLGeneratorAgent.generate() succeeds and the artifact file exists on disk.

NOTE on security (Phase 3 target):
  Current state: SHA-256 integrity check (detects corruption).
  Phase 3 target: asymmetric signing (ECDSA/RSA) over manifest.json.
  This node will be updated in Phase 3 when the signing infrastructure
  is implemented.
"""
import logging
import os
from datetime import datetime, timezone

from langchain_core.messages import SystemMessage, HumanMessage

from src.agentic_ml.state.agent_state import AgentState
from src.agentic_ml.ml_engine.pipelines.artifact_pipeline import PKLGeneratorAgent

logger = logging.getLogger("agentic_ml.agents.deployment")

SYSTEM_PROMPT = (
    "You are the Deployment Agent. "
    "Package and verify the production artifact bundle with SHA-256 integrity hashing. "
    "Confirm artifact size, version, and readiness for serving."
)


def deployment_node(state: AgentState) -> dict:
    from src.agentic_ml.llm.factory import get_llm
    llm = get_llm()

    best_name = state.get("best_model_name") or "RandomForest"
    trained_models = state.get("trained_models") or {}
    best_model = trained_models.get(best_name)

    if best_model is None:
        raise RuntimeError(
            f"Deployment node: best_model '{best_name}' not found in trained_models — "
            "deployment_completed NOT set."
        )

    generator = PKLGeneratorAgent()
    gen_result = generator.generate(
        pipeline_or_model=best_model,
        task=state.get("task_type", "classification"),
        model_name=best_name,
        feature_columns=state.get("selected_features"),
        target_column=state.get("target_column", "target"),
        metrics=state.get("best_model_metrics", {}),
        description=f"Automated build of {best_name} for task {state.get('task_type')}",
        register_version=True,
    )

    artifact_path = gen_result["filepath"]
    sha256_hash = gen_result.get("sha256", "")

    # Verify artifact actually exists on disk before declaring completion.
    if not os.path.exists(artifact_path):
        raise RuntimeError(
            f"Artifact file not found at {artifact_path} — deployment_completed NOT set."
        )

    logger.info(
        "Deployment: artifact saved → %s (%d bytes, SHA-256: %s...).",
        artifact_path, gen_result.get("size_bytes", 0), sha256_hash[:12]
    )

    execution_mode = state.get("execution_mode", "simulation")
    try:
        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Production artifact created: {artifact_path} "
                    f"(SHA-256: {sha256_hash[:12]}...). "
                    f"Size: {gen_result.get('size_bytes', 0)} bytes."
                )
            ),
        ])
        execution_mode = "live"
    except Exception as exc:
        from langchain_core.messages import AIMessage
        response = AIMessage(
            content=(
                f"[Deployment — Simulation Mode]\n"
                f"Artifact: {artifact_path}\n"
                f"SHA-256: {sha256_hash[:12]}...\n"
                f"LLM unavailable: {exc}"
            )
        )
        logger.warning("Deployment: simulation fallback — %s", exc)

    provenance_entry = {
        "agent_name": "deployment",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "operation": "PKL artifact bundle generation with SHA-256 integrity hash",
        "result_summary": f"model={best_name}, size={gen_result.get('size_bytes', 0)}, sha256={sha256_hash[:16]}",
        "artifact_path": artifact_path,
    }

    return {
        "messages": [response],
        "artifact_path": artifact_path,
        "deployment_completed": True,
        "execution_mode": execution_mode,
        "provenance": [provenance_entry],
    }
