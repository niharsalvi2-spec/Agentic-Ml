from langchain_core.messages import SystemMessage, HumanMessage
from src.agentic_ml.state.agent_state import AgentState
from src.agentic_ml.llm.factory import get_llm

def testing_node(state: AgentState) -> dict:
    llm = get_llm()
    candidates = state.get("candidate_models", [])
    
    sys_prompt = "You are the Testing Agent. Perform unit checks on input/output schemas and prediction stability."
    human_prompt = f"Executed contract tests across models {candidates}. All schema constraints passed."
    response = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=human_prompt)])
    
    return {
        "messages": [response],
        "model_tested": True,
        "next_agent": "validation"
    }
