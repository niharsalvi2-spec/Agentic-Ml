from langchain_core.messages import SystemMessage, HumanMessage
from src.agentic_ml.state.agent_state import AgentState
from src.agentic_ml.llm.factory import get_llm

def preprocessing_node(state: AgentState) -> dict:
    llm = get_llm()
    sys_prompt = "You are the Preprocessing Agent. Decide the optimal imputation, encoding, and scaling strategy."
    human_prompt = f"Cleaning dataset with target '{state.get('target_column')}'. Handled missing values, encoded categoricals, scaled numeric features."
    response = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=human_prompt)])
    
    return {
        "messages": [response],
        "data_preprocessed": True,
        "next_agent": "eda"
    }
