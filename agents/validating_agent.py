from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from .state import AgentState
from .llm_factory import get_llm

LOGIC_MATRIX = """
You are the Validating Agent.
Check for the 7 common evaluation mistakes:
1. Evaluating on training data.
2. Using accuracy for imbalanced data.
3. Optimizing threshold on test set.
4. Ignoring class costs.
5. Relying on a single metric.
6. Data leakage in preprocessing.
7. Overfitting (compare train vs test).
"""

def validating_agent_node(state: AgentState, config: RunnableConfig):
    llm = get_llm()
    messages = state.get('messages', [])
    last_message = messages[-1].content if messages else "No input"

    print(f"[Validating Agent] Auditing model and pipeline for common mistakes...")
    
    sys_msg = SystemMessage(content=LOGIC_MATRIX)
    human_msg = HumanMessage(content=f"Current context/request: {last_message}")
    response = llm.invoke([sys_msg, human_msg])
    
    new_state = {
        "messages": [response],
        "model_validated": True,
        "next_agent": "deploying_agent"
    }
    
    return new_state
