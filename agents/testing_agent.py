from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from .state import AgentState
from .llm_factory import get_llm

LOGIC_MATRIX = """
You are the Testing Agent.
Evaluate the model based on specific problem type and class dynamics:
- Classification: Balanced (Accuracy + F1), Imbalanced (Recall, PR-AUC, Precision, F-Beta).
- Regression: MAE (robust), RMSE (penalizes large errors), R2 (fit quality).
- Clustering: Silhouette, Davies-Bouldin, ARI, NMI.
"""

def testing_agent_node(state: AgentState, config: RunnableConfig):
    llm = get_llm()
    messages = state.get('messages', [])
    last_message = messages[-1].content if messages else "No input"

    print(f"[Testing Agent] Evaluating model performance...")
    
    sys_msg = SystemMessage(content=LOGIC_MATRIX)
    human_msg = HumanMessage(content=f"Current context/request: {last_message}")
    response = llm.invoke([sys_msg, human_msg])
    
    new_state = {
        "messages": [response],
        "model_tested": True,
        "next_agent": "validating_agent"
    }
    
    return new_state
