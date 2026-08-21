import sys
from pathlib import Path
from langchain_core.messages import HumanMessage

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(root_dir))

from src.agentic_ml.orchestration.graph import build_agentic_graph

def main():
    print("==========================================================")
    print("[*] Launching Autonomous Agentic ML Engineer System...")
    print("==========================================================\n")
    
    app = build_agentic_graph()
    
    initial_state = {
        "messages": [HumanMessage(content="I want an optimal model to predict Customer Churn.")],
        "raw_prompt": "I want an optimal model to predict Customer Churn.",
        "current_task": "Predict Customer Churn",
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
    
    print("Executing 10-Agent Lifecycle Graph:\n")
    for output in app.stream(initial_state):
        for node_name, state_update in output.items():
            print(f"--> [{node_name.upper()} COMPLETED]")
            if "messages" in state_update and state_update["messages"]:
                print(f"   {state_update['messages'][-1].content}\n")
                
    print("==========================================================")
    print("[+] Pipeline Completed. Model Artifact `model.pkl` Ready.")
    print("==========================================================")

if __name__ == "__main__":
    main()
