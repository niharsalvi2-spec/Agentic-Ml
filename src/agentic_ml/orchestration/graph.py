"""
LangGraph orchestration graph for the Agentic ML Engineering Platform.

Routing architecture (per locked design decision):
  - Each agent node returns a LangGraph Command(goto=..., update={...}) that explicitly
    names the next node. The router is the agent, not a separate master_router function.
  - AgentState carries completion evidence; Command carries routing intent.
  - There is NO master_router, NO next_agent state field, NO dual-authority.

Pipeline sequence:
  START → problem_analyzer → data_collector → preprocessing → eda
        → feature_engineering → feature_selection → model_building
        → testing → validation → deployment → END

Error paths: any agent returning Command(goto=END) short-circuits the pipeline safely.
"""

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


PIPELINE_SEQUENCE = [
    ("problem_analyzer",    problem_analyzer_node),
    ("data_collector",      data_collector_node),
    ("preprocessing",       preprocessing_node),
    ("eda",                 eda_node),
    ("feature_engineering", feature_engineering_node),
    ("feature_selection",   feature_selection_node),
    ("model_building",      model_building_node),
    ("testing",             testing_node),
    ("validation",          validation_node),
    ("deployment",          deployment_node),
]


def build_agentic_graph():
    """
    Compile and return the LangGraph StateGraph for the ML pipeline.

    Each node is an agent function that returns a LangGraph Command object declaring
    both its state updates and the explicit destination node (`goto`).

    Returns:
        CompiledGraph: The compiled LangGraph application ready for .stream() or .invoke().
    """
    workflow = StateGraph(AgentState)

    # Register all agent nodes.
    for name, node_fn in PIPELINE_SEQUENCE:
        workflow.add_node(name, node_fn)

    # Wire the initial entry point to the first node in the sequence.
    # Subsequent transitions are exclusively driven by Command(goto=...) returns.
    workflow.add_edge(START, "problem_analyzer")

    return workflow.compile()
