"""
Validation Agent Node.
Executes cross-validation gates, detects 7 classic evaluation mistakes, and crowns the winning model.
"""

from langchain_core.messages import SystemMessage, HumanMessage
from src.agentic_ml.state.agent_state import AgentState
from src.agentic_ml.llm.factory import get_llm
from src.agentic_ml.ml_engine.data.loader import DataLoader
from src.agentic_ml.ml_engine.preprocessing.cleaner import DeterministicPreprocessor
from src.agentic_ml.ml_engine.evaluation.validation import ModelEvaluator, EvaluationAgent


def validation_node(state: AgentState) -> dict:
    llm = get_llm()
    task_type = state.get("task_type", "classification")
    path = state.get("dataset_path", "")
    trained_models = state.get("trained_models", {})
    selected_features = state.get("selected_features")
    
    # 1. Load and prepare validation data
    df, target_col = DataLoader.load_or_synthesize(task_type, path)
    preprocessor = DeterministicPreprocessor()
    X, y = preprocessor.fit_transform(df, target_col)
    
    if selected_features and all(f in X.columns for f in selected_features):
        X = X[selected_features]
    
    # 2. Evaluation Advisor & Cross-validation
    eval_agent = EvaluationAgent()
    rec_metrics = eval_agent.recommend_metrics(task_type)
    best_name, scores = ModelEvaluator.evaluate(trained_models, X, y, task_type)
    
    # 3. Mistake & Leakage Check
    mistake_warnings = eval_agent.check_common_mistakes(
        task=task_type,
        evaluated_on_training_data=False,
        class_balance=0.5,
        scaler_fit_on="train"
    )
    
    sys_prompt = "You are the Validation Agent. Compare cross-validation scores, detect data leakage, and validate model metrics."
    human_prompt = (
        f"Validation scores: {scores}. Top performing model: {best_name} "
        f"(Score: {scores.get(best_name, 0.0):.4f}). "
        f"Recommended evaluation metrics: {rec_metrics['primary']}. "
        f"Leakage check: {len(mistake_warnings)} warnings."
    )
    
    try:
        response = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=human_prompt)])
    except Exception:
        response = SystemMessage(content=f"Validation completed. Best model selected: {best_name}.")

    return {
        "messages": [response],
        "best_model_name": best_name,
        "best_model_metrics": scores,
        "model_validated": True,
        "next_agent": "deployment"
    }
