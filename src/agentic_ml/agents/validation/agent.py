from langchain_core.messages import SystemMessage, HumanMessage
from src.agentic_ml.state.agent_state import AgentState
from src.agentic_ml.llm.factory import get_llm
from src.agentic_ml.ml_engine.data.loader import DataLoader
from src.agentic_ml.ml_engine.preprocessing.cleaner import DeterministicPreprocessor
from src.agentic_ml.ml_engine.evaluation.validation import ModelEvaluator

def validation_node(state: AgentState) -> dict:
    llm = get_llm()
    task_type = state.get("task_type", "classification")
    path = state.get("dataset_path", "")
    trained_models = state.get("trained_models", {})
    
    # 1. Load and prepare validation data
    df, target_col = DataLoader.load_or_synthesize(task_type, path)
    preprocessor = DeterministicPreprocessor()
    X, y = preprocessor.fit_transform(df, target_col)
    
    # 2. Evaluate via cross-validation gate
    best_name, scores = ModelEvaluator.evaluate(trained_models, X, y, task_type)
    
    sys_prompt = "You are the Validation Agent. Compare k-fold scores, detect data leakage, and crown the best model."
    human_prompt = f"Validation scores: {scores}. Top performing model: {best_name} (Score: {scores.get(best_name, 0.0):.4f})."
    response = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=human_prompt)])
    
    return {
        "messages": [response],
        "best_model_name": best_name,
        "best_model_metrics": scores,
        "model_validated": True,
        "next_agent": "deployment"
    }
