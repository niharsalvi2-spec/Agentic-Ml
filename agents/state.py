from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """The state of the ML Engineer orchestrator graph."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    
    # Task context
    current_task: str
    dataset_path: str
    dataset_info: dict
    
    # Flags and progress
    data_engineering_completed: bool
    data_collected: bool
    data_preprocessed: bool
    eda_completed: bool
    feature_selection_completed: bool
    model_built: bool
    model_tested: bool
    model_validated: bool
    deployment_completed: bool
    
    # Next agent to route to
    next_agent: str
