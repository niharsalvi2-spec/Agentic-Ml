"""
Feature Engineering Agent Node.
Constructs polynomial interactions, normalized ratios, and log transforms for skewed features.
"""

from langchain_core.messages import SystemMessage, HumanMessage
from src.agentic_ml.state.agent_state import AgentState
from src.agentic_ml.llm.factory import get_llm
from src.agentic_ml.ml_engine.data.loader import DataLoader
from src.agentic_ml.ml_engine.features.engineering import FeatureEngineer


def feature_engineering_node(state: AgentState) -> dict:
    llm = get_llm()
    task_type = state.get("task_type", "classification")
    path = state.get("dataset_path", "")
    
    # 1. Load data
    df, target_col = DataLoader.load_or_synthesize(task_type, path)
    X = df.drop(columns=[target_col], errors="ignore")
    
    # 2. Construct engineering transformations
    X_log = FeatureEngineer.add_log_transforms(X)
    X_engineered = FeatureEngineer.add_polynomial_interactions(X_log, max_features=4)
    
    added_features = [c for c in X_engineered.columns if c not in X.columns]
    
    sys_prompt = "You are the Feature Engineering Agent. Formulate domain interaction features, ratios, and transformations."
    human_prompt = f"Engineered {len(added_features)} new feature representations: {added_features[:5]}."
    
    try:
        response = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=human_prompt)])
    except Exception:
        response = SystemMessage(content=f"Feature engineering constructed {len(added_features)} candidate features.")
        
    return {
        "messages": [response],
        "feature_engineered": True,
        "next_agent": "feature_selection"
    }
