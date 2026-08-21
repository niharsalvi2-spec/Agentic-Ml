from langchain_core.messages import SystemMessage, HumanMessage
from src.agentic_ml.state.agent_state import AgentState
from src.agentic_ml.llm.factory import get_llm

def problem_analyzer_node(state: AgentState) -> dict:
    llm = get_llm()
    prompt = state.get("raw_prompt") or state.get("current_task", "Build an optimal ML model.")
    
    # Infer task type
    task_type = "classification" if any(w in prompt.lower() for w in ["churn", "classify", "spam", "fraud", "detect"]) else "regression"
    
    sys_prompt = "You are the Problem Analyzer Agent. Analyze the ML problem statement, determine the task type, metrics, and goals."
    human_prompt = f"Problem statement: {prompt}"
    response = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=human_prompt)])
    
    return {
        "messages": [response],
        "task_type": task_type,
        "problem_analyzed": True,
        "next_agent": "data_collector"
    }
