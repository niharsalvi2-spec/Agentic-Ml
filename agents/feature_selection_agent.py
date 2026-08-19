from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from .state import AgentState
from .llm_factory import get_llm

LOGIC_MATRIX = """
You are the Feature Selection Engineer.
Apply the following encoding strategies based on cardinality: 
Binary -> Label encode. Nominal (low) -> One-hot. Nominal (med/high) -> Target/Frequency encoding.
Apply Feature Extraction (PCA/t-SNE/Autoencoders) and Feature Selection (Filter/Wrapper/Embedded methods).
"""

def feature_selection_agent_node(state: AgentState, config: RunnableConfig):
    llm = get_llm()
    messages = state.get('messages', [])
    last_message = messages[-1].content if messages else "No input"

    print(f"[Feature Selection Engineer] Engineering and selecting features...")
    
    sys_msg = SystemMessage(content=LOGIC_MATRIX)
    human_msg = HumanMessage(content=f"Current context/request: {last_message}")
    response = llm.invoke([sys_msg, human_msg])
    
    new_state = {
        "messages": [response],
        "feature_selection_completed": True,
        "next_agent": "model_building"
    }
    
    return new_state
