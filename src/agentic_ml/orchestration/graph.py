from langgraph.graph import StateGraph, START, END
from src.agentic_ml.state.agent_state import AgentState
from src.agentic_ml.agents.problem_analyzer.agent import problem_analyzer_node
from src.agentic_ml.agents.data_collector.agent import data_collector_node
from src.agentic_ml.agents.preprocessing.agent import preprocessing_node
from src.agentic_ml.agents.eda.agent import eda_node
from src.agentic_ml.agents.feature_engineering.agent import feature_engineering_node
from src.agentic_ml.agents.feature_selection.agent import feature_selection_node
from src.agentic_ml.agents.model_building.agent import model_building_node
from src.agentic_ml.agents.testing.agent import testing_node
from src.agentic_ml.agents.validation.agent import validation_node
from src.agentic_ml.agents.deployment.agent import deployment_node

def master_router(state: AgentState):
    next_node = state.get("next_agent")
    if next_node:
        return next_node
        
    if not state.get("problem_analyzed"):
        return "problem_analyzer"
    if not state.get("data_collected"):
        return "data_collector"
    if not state.get("data_preprocessed"):
        return "preprocessing"
    if not state.get("eda_completed"):
        return "eda"
    if not state.get("feature_engineered"):
        return "feature_engineering"
    if not state.get("feature_selection_completed"):
        return "feature_selection"
    if not state.get("model_built"):
        return "model_building"
    if not state.get("model_tested"):
        return "testing"
    if not state.get("model_validated"):
        return "validation"
    if not state.get("deployment_completed"):
        return "deployment"
        
    return END

def build_agentic_graph():
    workflow = StateGraph(AgentState)
    
    nodes = {
        "problem_analyzer": problem_analyzer_node,
        "data_collector": data_collector_node,
        "preprocessing": preprocessing_node,
        "eda": eda_node,
        "feature_engineering": feature_engineering_node,
        "feature_selection": feature_selection_node,
        "model_building": model_building_node,
        "testing": testing_node,
        "validation": validation_node,
        "deployment": deployment_node
    }
    
    for name, node in nodes.items():
        workflow.add_node(name, node)
        
    node_names = list(nodes.keys())
    route_map = {n: n for n in node_names} | {END: END, "END": END}
    
    workflow.add_conditional_edges(START, master_router, route_map)
    for name in node_names:
        workflow.add_conditional_edges(name, master_router, route_map)
        
    return workflow.compile()
