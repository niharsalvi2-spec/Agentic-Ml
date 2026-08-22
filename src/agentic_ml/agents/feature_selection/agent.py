"""
Feature Selection Agent Node.
Performs statistical ANOVA, Mutual Info, and Tree Importance feature selection.
"""

from langchain_core.messages import SystemMessage, HumanMessage
from src.agentic_ml.state.agent_state import AgentState
from src.agentic_ml.llm.factory import get_llm
from src.agentic_ml.ml_engine.data.loader import DataLoader
from src.agentic_ml.ml_engine.preprocessing.cleaner import DeterministicPreprocessor
from src.agentic_ml.ml_engine.features.selection import FeatureSelector


def feature_selection_node(state: AgentState) -> dict:
    llm = get_llm()
    task_type = state.get("task_type", "classification")
    path = state.get("dataset_path", "")
    
    # 1. Load & clean
    df, target_col = DataLoader.load_or_synthesize(task_type, path)
    preprocessor = DeterministicPreprocessor()
    X, y = preprocessor.fit_transform(df, target_col)
    
    # 2. Select top features
    k = min(4, X.shape[1])
    selected = FeatureSelector.select_top_k(X, y, task_type=task_type, k=k)
    
    sys_prompt = "You are the Feature Selection Agent. Filter out redundant noise features based on ANOVA / mutual info."
    human_prompt = f"Evaluated {X.shape[1]} candidate features. Retained top {len(selected)} high-signal features: {selected}."
    
    try:
        response = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=human_prompt)])
    except Exception:
        response = SystemMessage(content=f"Feature selection retained top {len(selected)} features: {selected}.")

    return {
        "messages": [response],
        "selected_features": selected,
        "feature_selection_completed": True,
        "next_agent": "model_building"
    }
