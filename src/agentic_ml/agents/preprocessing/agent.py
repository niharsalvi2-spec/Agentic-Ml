"""
Preprocessing Agent Node.
Executes leakage-safe missing value imputation, outlier fence boundaries, and categorical encoding.
"""

from langchain_core.messages import SystemMessage, HumanMessage
from src.agentic_ml.state.agent_state import AgentState
from src.agentic_ml.llm.factory import get_llm
from src.agentic_ml.ml_engine.data.loader import DataLoader
from src.agentic_ml.ml_engine.preprocessing.cleaner import DeterministicPreprocessor, missingness_report


def preprocessing_node(state: AgentState) -> dict:
    llm = get_llm()
    task_type = state.get("task_type", "classification")
    path = state.get("dataset_path", "")
    
    # 1. Load data
    df, target_col = DataLoader.load_or_synthesize(task_type, path)
    
    # 2. Compute missingness diagnostics
    miss_rep = missingness_report(df).to_dict(orient="index")
    
    # 3. Deterministic Preprocessing
    preprocessor = DeterministicPreprocessor(clip_outliers=True)
    X, y = preprocessor.fit_transform(df, target_col)
    
    sys_prompt = "You are the Preprocessing Agent. Report on missingness diagnostics, outlier fence clipping, and feature scaling."
    human_prompt = (
        f"Preprocessed dataset with target '{target_col}'. "
        f"Input shape: {df.shape} -> Processed feature matrix: {X.shape}. "
        f"Missing values handled across {len(miss_rep)} columns."
    )
    
    try:
        response = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=human_prompt)])
    except Exception:
        response = SystemMessage(content="Preprocessing completed deterministically.")

    return {
        "messages": [response],
        "target_column": target_col,
        "data_preprocessed": True,
        "next_agent": "eda"
    }
