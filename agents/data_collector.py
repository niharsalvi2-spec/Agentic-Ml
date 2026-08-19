from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from .state import AgentState
from .llm_factory import get_llm

LOGIC_MATRIX = """
Data Collector Agent: Responsible for identifying and gathering data.

Decision Logic based on user request:
1. Already exists publicly: Web Scraping, APIs, Kaggle.
2. Exists but restricted: Data Marketplace, Synthetic Data, Crowdsourcing.
"""

def data_collector_node(state: AgentState, config: RunnableConfig):
    llm = get_llm()
    messages = state.get('messages', [])
    last_message = messages[-1].content if messages else "No input"

    print(f"[Data Collector Agent] Analyzing request...")
    
    sys_msg = SystemMessage(content=LOGIC_MATRIX)
    human_msg = HumanMessage(content=f"Current context/request: {last_message}")
    response = llm.invoke([sys_msg, human_msg])
    
    new_state = {
        "messages": [response],
        "data_collected": True,
        "next_agent": "data_preprocessor"
    }
    
    return new_state
