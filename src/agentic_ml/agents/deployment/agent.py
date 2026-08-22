"""
Deployment Agent Node.
Packages the winning model into a secure, self-describing, SHA-256 hash-verified production .pkl bundle.
"""

from langchain_core.messages import SystemMessage, HumanMessage
from src.agentic_ml.state.agent_state import AgentState
from src.agentic_ml.llm.factory import get_llm
from src.agentic_ml.ml_engine.pipelines.artifact_pipeline import PKLGeneratorAgent


def deployment_node(state: AgentState) -> dict:
    llm = get_llm()
    best_name = state.get("best_model_name", "RandomForest")
    trained_models = state.get("trained_models", {})
    best_model = trained_models.get(best_name)
    
    generator = PKLGeneratorAgent()
    
    # Assembly of self-describing bundle with integrity hash
    gen_result = generator.generate(
        pipeline_or_model=best_model,
        task=state.get("task_type", "classification"),
        model_name=best_name,
        feature_columns=state.get("selected_features"),
        target_column=state.get("target_column", "target"),
        metrics=state.get("best_model_metrics", {}),
        description=f"Automated build of {best_name} for task {state.get('task_type')}",
        register_version=True
    )
    
    artifact_path = gen_result["filepath"]
    sha256_hash = gen_result.get("sha256", "")
    
    sys_prompt = "You are the Deployment Agent. Package and verify the production artifact bundle with integrity hashing."
    human_prompt = f"Production artifact created: {artifact_path} (SHA-256: {sha256_hash[:12]}...). Size: {gen_result['size_bytes']} bytes."
    
    try:
        response = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=human_prompt)])
    except Exception:
        response = SystemMessage(content=f"Production artifact saved to {artifact_path}.")

    return {
        "messages": [response],
        "artifact_path": artifact_path,
        "deployment_completed": True,
        "next_agent": "END"
    }
