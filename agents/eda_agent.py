from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from .state import AgentState
from .llm_factory import get_llm

LOGIC_MATRIX = """
You are the Exploratory Data Analysis (EDA) Agent.
Analyze the request and decide the appropriate visualizations: Target vs Features, Distribution, Relationships, etc.
"""

def eda_agent_node(state: AgentState, config: RunnableConfig):
    llm = get_llm()
    messages = state.get('messages', [])
    last_message = messages[-1].content if messages else "No input"

    print(f"[EDA Agent] Performing Exploratory Data Analysis...")
    
    sys_msg = SystemMessage(content=LOGIC_MATRIX)
    human_msg = HumanMessage(content=f"Current context/request: {last_message}")
    response = llm.invoke([sys_msg, human_msg])
    
    new_state = {
        "messages": [response],
        "eda_completed": True,
        "next_agent": "feature_selection"
    }
    
    return new_state
