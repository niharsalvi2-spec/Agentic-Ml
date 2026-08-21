from src.agentic_ml.orchestration.graph import build_agentic_graph
from langchain_core.messages import HumanMessage

def mcp_run_pipeline(task: str) -> str:
    app = build_agentic_graph()
    initial_state = {
        "messages": [HumanMessage(content=task)],
        "raw_prompt": task,
        "current_task": task,
        "task_type": "classification",
        "target_column": None,
        "dataset_path": "",
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
        "next_agent": None
    }
    log = []
    for output in app.stream(initial_state):
        for node, val in output.items():
            log.append(f"[{node.upper()}] completed")
    return "\n".join(log)
