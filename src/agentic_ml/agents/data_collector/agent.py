from langchain_core.messages import SystemMessage, HumanMessage
from src.agentic_ml.state.agent_state import AgentState
from src.agentic_ml.llm.factory import get_llm
from src.agentic_ml.ml_engine.data.loader import DataLoader
from src.agentic_ml.ml_engine.data.profiler import DataProfiler

def data_collector_node(state: AgentState) -> dict:
    llm = get_llm()
    task_type = state.get("task_type", "classification")
    path = state.get("dataset_path", "")
    
    # Execute deterministic load/synthesis
    df, target_col = DataLoader.load_or_synthesize(task_type, path)
    profile = DataProfiler.profile(df, target_col)
    
    sys_prompt = "You are the Data Collector Agent. Summarize dataset acquisition and schema properties."
    human_prompt = f"Dataset profiled: {profile['n_rows']} rows, {profile['n_columns']} cols. Target: {target_col}"
    response = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=human_prompt)])
    
    return {
        "messages": [response],
        "target_column": target_col,
        "dataset_info": profile,
        "data_collected": True,
        "next_agent": "preprocessing"
    }
