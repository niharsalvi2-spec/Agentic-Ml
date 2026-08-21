from typing import TypedDict, Annotated, Sequence, Optional, Dict, Any, List
import operator
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """The central state of the ML Engineer orchestrator graph."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    
    # Task Context
    raw_prompt: str
    current_task: str
    task_type: str  # classification, regression, clustering
    target_column: Optional[str]
    
    # Data Context
    dataset_path: str
    dataset_info: Dict[str, Any]
    data_summary: Dict[str, Any]
    
    # Feature & Model Context
    selected_features: List[str]
    candidate_models: List[str]
    trained_models: Dict[str, Any]
    best_model_name: Optional[str]
    best_model_metrics: Dict[str, float]
    
    # Artifact context
    artifact_path: Optional[str]
    
    # Pipeline Progression Flags
    problem_analyzed: bool
    data_collected: bool
    data_preprocessed: bool
    eda_completed: bool
    feature_engineered: bool
    feature_selection_completed: bool
    model_built: bool
    model_tested: bool
    model_validated: bool
    deployment_completed: bool
    
    # Dynamic Routing
    next_agent: Optional[str]
