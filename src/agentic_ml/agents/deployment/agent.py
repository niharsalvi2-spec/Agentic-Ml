"""
Deployment Agent Node.

Packages the winning model into a secure, self-describing, asymmetrically signed
production artifact bundle. Sets deployment_completed=True only after
ArtifactBundleManager creates the versioned bundle directory with signed manifest.json.
Transitions to END via LangGraph Command.
"""
import logging
import os
from datetime import datetime, timezone

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.types import Command
from langgraph.graph import END

from src.agentic_ml.state.agent_state import AgentState
from src.agentic_ml.llm.factory import get_llm
from src.agentic_ml.security.manifest import ArtifactBundleManager
from src.agentic_ml.ml_engine.pipelines.artifact_pipeline import PKLGeneratorAgent

logger = logging.getLogger("agentic_ml.agents.deployment")

SYSTEM_PROMPT = (
    "You are the Deployment Agent. "
    "Package and verify the production artifact bundle with asymmetric ECDSA signing and SHA-256 manifest integrity. "
    "Confirm artifact size, version, and readiness for serving."
)


def deployment_node(state: AgentState) -> Command:
    llm = get_llm()
    best_name = state.get("best_model_name") or "RandomForest"
    trained_models = state.get("trained_models") or {}
    best_model = trained_models.get(best_name)

    if best_model is None:
        raise RuntimeError(
            f"Deployment node: best_model '{best_name}' not found in trained_models — "
            "deployment_completed NOT set."
        )

    # 1. Generate standard PKL file for backward compatibility
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

    # 2. Create asymmetrically signed artifact bundle
    bundle_info = ArtifactBundleManager.create_bundle(
        model_name=best_name,
        model_obj=best_model,
        task_type=state.get("task_type", "classification"),
        feature_columns=state.get("selected_features"),
        target_column=state.get("target_column", "target"),
        metrics=state.get("best_model_metrics", {}),
        provenance=state.get("provenance", []),
        description=f"Automated build of {best_name} for task {state.get('task_type')}",
    )

    artifact_path = bundle_info["bundle_dir"]
    model_sha256 = bundle_info["hashes"].get("model.pkl", "")

    # Verify bundle directory and files exist on disk before declaring completion
    if not os.path.exists(bundle_info["manifest_path"]) or not os.path.exists(bundle_info["signature_path"]):
        raise RuntimeError(
            f"Artifact bundle manifest/signature not found at {artifact_path} — deployment_completed NOT set."
        )

    logger.info(
        "Deployment: artifact bundle saved → %s (Version: %s, SHA-256: %s...).",
        artifact_path, bundle_info["version"], model_sha256[:12]
    )

    execution_mode = state.get("execution_mode", "simulation")
    try:
        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Production artifact bundle created: {artifact_path} "
                    f"[Version: {bundle_info['version']}, SHA-256: {model_sha256[:12]}...]. "
                    f"Digitally signed with ECDSA-SHA256."
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
                f"SHA-256: {model_sha256[:12]}...\n"
                f"LLM unavailable: {exc}"
            )
        )
        logger.warning("Deployment: simulation fallback — %s", exc)

    provenance_entry = {
        "agent_name": "deployment",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "operation": "Asymmetric ECDSA-signed artifact bundle generation",
        "result_summary": f"model={best_name}, version={bundle_info['version']}, sha256={model_sha256[:16]}",
        "artifact_path": artifact_path,
    }

    return Command(
        goto=END,
        update={
            "messages": [response],
            "artifact_path": artifact_path,
            "deployment_completed": True,
            "execution_mode": execution_mode,
            "provenance": [provenance_entry],
        },
    )
