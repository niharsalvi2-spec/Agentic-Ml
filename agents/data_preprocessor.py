from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from .state import AgentState
from .llm_factory import get_llm

LOGIC_MATRIX = """
You are the Data Preprocessor Agent.
Responsibilities:
1. Detect and Handle Missing Values (MCAR/MAR/MNAR).
2. Remove/Handle Duplicates.
3. Detect and handle Outliers (Z-score, IQR capping).
4. Perform Anomaly detection (Isolation Forest).
Analyze the dataset constraints and return your preprocessing plan.
"""

def data_preprocessor_node(state: AgentState, config: RunnableConfig):
    llm = get_llm()
    messages = state.get('messages', [])
    last_message = messages[-1].content if messages else "No input"

    print(f"[Data Preprocessor Agent] Processing data...")
    
    sys_msg = SystemMessage(content=LOGIC_MATRIX)
    human_msg = HumanMessage(content=f"Current context/request: {last_message}")
    response = llm.invoke([sys_msg, human_msg])
    
    new_state = {
        "messages": [response],
        "data_preprocessed": True,
        "next_agent": "eda_agent"
    }
    
    return new_state
