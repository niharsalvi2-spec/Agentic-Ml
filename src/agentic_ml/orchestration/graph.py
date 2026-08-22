"""
LangGraph orchestration graph for the Agentic ML Engineering Platform.

Routing architecture (per design decision):
  - Each agent node returns a LangGraph Command that explicitly names the
    next node. The router is the agent, not a separate master_router function.
  - AgentState carries completion evidence; Command carries routing intent.
  - There is NO master_router, NO next_agent state field, NO dual-authority.

Pipeline sequence (fixed sequential order, Phase 1):
  START → problem_analyzer → data_collector → preprocessing → eda
        → feature_engineering → feature_selection → model_building
        → testing → validation → deployment → END

Error paths: any agent that returns Command(goto=END) short-circuits the
pipeline. The full error is written into the agent's provenance entry.
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


# Ordered pipeline sequence — single source of truth for execution order.
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

    Each node is an agent function. Agents return LangGraph Command objects
    that explicitly declare the next node. There is no shared master_router.

    Returns:
        CompiledGraph: The compiled LangGraph application ready for .stream()
                       or .invoke().
    """
    workflow = StateGraph(AgentState)

    # Register all agent nodes.
    for name, node_fn in PIPELINE_SEQUENCE:
        workflow.add_node(name, node_fn)

    # Wire the sequential edges (START → first agent, last agent → END).
    # Individual agents control branching via Command(goto=...).
    node_names = [name for name, _ in PIPELINE_SEQUENCE]
    workflow.add_edge(START, node_names[0])
    for i in range(len(node_names) - 1):
        workflow.add_edge(node_names[i], node_names[i + 1])
    workflow.add_edge(node_names[-1], END)

    return workflow.compile()
