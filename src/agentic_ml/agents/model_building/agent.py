from langchain_core.messages import SystemMessage, HumanMessage
from src.agentic_ml.state.agent_state import AgentState
from src.agentic_ml.llm.factory import get_llm
from src.agentic_ml.ml_engine.data.loader import DataLoader
from src.agentic_ml.ml_engine.preprocessing.cleaner import DeterministicPreprocessor
from src.agentic_ml.ml_engine.models.training import ModelTrainer

def model_building_node(state: AgentState) -> dict:
    llm = get_llm()
    task_type = state.get("task_type", "classification")
    path = state.get("dataset_path", "")
    
    # 1. Load data
    df, target_col = DataLoader.load_or_synthesize(task_type, path)
    
    # 2. Preprocess
    preprocessor = DeterministicPreprocessor()
    X, y = preprocessor.fit_transform(df, target_col)
    
    # 3. Train candidate models deterministically
    trained_models = ModelTrainer.train_candidates(X, y, task_type)
    candidates = list(trained_models.keys())
    
    sys_prompt = "You are the Model Building Agent. Train diverse algorithm families and optimize architectures."
    human_prompt = f"Trained {len(candidates)} baseline model candidates: {candidates}."
    response = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=human_prompt)])
    
    return {
        "messages": [response],
        "candidate_models": candidates,
        "trained_models": trained_models,
        "model_built": True,
        "next_agent": "testing"
    }
