from langchain_core.messages import SystemMessage, HumanMessage
from src.agentic_ml.state.agent_state import AgentState
from src.agentic_ml.llm.factory import get_llm
from src.agentic_ml.ml_engine.pipelines.artifact_pipeline import ArtifactSerializer

def deployment_node(state: AgentState) -> dict:
    llm = get_llm()
    best_name = state.get("best_model_name", "RandomForest")
    trained_models = state.get("trained_models", {})
    best_model = trained_models.get(best_name)
    
    # Package into model.pkl artifact
    metadata = {
        "model_name": best_name,
        "task_type": state.get("task_type", "classification"),
        "target_column": state.get("target_column", "target"),
        "metrics": state.get("best_model_metrics", {}),
        "selected_features": state.get("selected_features", [])
    }
    
    artifact_path = ArtifactSerializer.save_artifact(best_model, metadata, filename="model.pkl")
    
    sys_prompt = "You are the Deployment Agent. Package and prepare the model.pkl production artifact."
    human_prompt = f"Production artifact built: {artifact_path}. Metrics: {metadata['metrics']}."
    response = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=human_prompt)])
    
    return {
        "messages": [response],
        "artifact_path": artifact_path,
        "deployment_completed": True,
        "next_agent": "END"
    }
