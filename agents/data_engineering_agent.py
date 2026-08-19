from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from .state import AgentState
from .llm_factory import get_llm

LOGIC_MATRIX = """
You are the Data Engineering & OLAP Architect Agent.
Analyze the request and decide the storage strategy (Data Warehouse, Data Lake, Lakehouse), schema (Star, Snowflake, Fact Constellation), and OLAP strategy (HOLAP, MOLAP, ROLAP).
"""

def data_engineering_agent_node(state: AgentState, config: RunnableConfig):
    llm = get_llm()
    messages = state.get('messages', [])
    last_message = messages[-1].content if messages else "No input"

    print(f"[Data Engineering Agent] Designing data infrastructure...")
    
    sys_msg = SystemMessage(content=LOGIC_MATRIX)
    human_msg = HumanMessage(content=f"Current context/request: {last_message}")
    response = llm.invoke([sys_msg, human_msg])
    
    new_state = {
        "messages": [response],
        "data_engineering_completed": True,
        "next_agent": "data_collector"
    }
    
    return new_state
