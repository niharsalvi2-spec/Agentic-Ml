from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from .state import AgentState
from .llm_factory import get_llm

LOGIC_MATRIX = """
You are the Model Building Agent.
Based on the task (Classification, Regression, Clustering) and dataset size, select the best model.
- Classification/Regression: Random Forest, XGBoost, LightGBM, CatBoost, SVM, Logistic/Linear Regression.
- Clustering: K-Means, DBSCAN, Hierarchical.
"""

def model_building_agent_node(state: AgentState, config: RunnableConfig):
    llm = get_llm()
    messages = state.get('messages', [])
    last_message = messages[-1].content if messages else "No input"

    print(f"[Model Building Agent] Selecting and training models...")
    
    sys_msg = SystemMessage(content=LOGIC_MATRIX)
    human_msg = HumanMessage(content=f"Current context/request: {last_message}")
    response = llm.invoke([sys_msg, human_msg])
    
    new_state = {
        "messages": [response],
        "model_built": True,
        "next_agent": "testing_agent"
    }
    
    return new_state
