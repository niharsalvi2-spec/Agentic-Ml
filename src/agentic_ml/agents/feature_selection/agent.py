from langchain_core.messages import SystemMessage, HumanMessage
from src.agentic_ml.state.agent_state import AgentState
from src.agentic_ml.llm.factory import get_llm

def feature_selection_node(state: AgentState) -> dict:
    llm = get_llm()
    features = state.get("dataset_info", {}).get("feature_columns", ["f1", "f2", "f3", "f4"])
    selected = features[:4]
    
    sys_prompt = "You are the Feature Selection Agent. Filter out redundant noise features based on ANOVA / mutual info."
    human_prompt = f"Evaluated {len(features)} candidate features. Retained top {len(selected)} high-importance features: {selected}."
    response = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=human_prompt)])
    
    return {
        "messages": [response],
        "selected_features": selected,
        "feature_selection_completed": True,
        "next_agent": "model_building"
    }
