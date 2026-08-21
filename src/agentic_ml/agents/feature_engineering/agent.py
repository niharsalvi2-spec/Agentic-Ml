from langchain_core.messages import SystemMessage, HumanMessage
from src.agentic_ml.state.agent_state import AgentState
from src.agentic_ml.llm.factory import get_llm

def feature_engineering_node(state: AgentState) -> dict:
    llm = get_llm()
    sys_prompt = "You are the Feature Engineering Agent. Formulate domain interaction features and ratios."
    human_prompt = "Engineered polynomial interaction features and normalized ratios across numeric dimensions."
    response = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=human_prompt)])
    
    return {
        "messages": [response],
        "feature_engineered": True,
        "next_agent": "feature_selection"
    }
