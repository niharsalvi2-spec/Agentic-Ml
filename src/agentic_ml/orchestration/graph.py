"""
LangGraph orchestration graph for the Agentic ML Engineering Platform.

Architecture:
  - Each agent node returns Command(goto=..., update={...}) — no master_router.
  - AgentState carries evidence; Command carries routing intent.
  - Checkpointer: SqliteSaver (file-based, suitable for dev/demo).
  - HITL: validation → deployment is gated by risk score.
    High-risk runs trigger an interrupt before "deployment".
    The API resumes via POST /api/pipeline/run/{run_id}/approve.

Graph topology:
  START
    → problem_analyzer
    → data_collector
    → preprocessing
    → eda
    → feature_engineering
    → feature_selection
    → model_building
    → testing
    → validation
      ├── PASS  → [risk gate] → deployment → END
      └── FAIL  → failure_analyzer
                   ├── retry  → model_building  (max 2 retries)
                   ├── preproc → preprocessing
                   └── stop   → END

The validation→failure_analyzer→model_building loop is what makes this
genuinely agentic: the graph can reason about failure and revise strategy.
"""

from typing import Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

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
from src.agentic_ml.agents.failure_analyzer.agent import failure_analyzer_node
from src.agentic_ml.agents.deployment.agent import deployment_node


def build_agentic_graph(checkpointer=None):
    """
    Compile and return the LangGraph StateGraph for the ML pipeline.

    Args:
        checkpointer: LangGraph checkpointer instance (SqliteSaver, MemorySaver, etc.)
                      If None, uses MemorySaver (in-process, no persistence).

    Returns:
        CompiledGraph: compiled LangGraph application ready for .stream() or .invoke().
    """
    workflow = StateGraph(AgentState)

    # ── Register all agent nodes ─────────────────────────────────────────────
    workflow.add_node("problem_analyzer",    problem_analyzer_node)
    workflow.add_node("data_collector",      data_collector_node)
    workflow.add_node("preprocessing",       preprocessing_node)
    workflow.add_node("eda",                 eda_node)
    workflow.add_node("feature_engineering", feature_engineering_node)
    workflow.add_node("feature_selection",   feature_selection_node)
    workflow.add_node("model_building",      model_building_node)
    workflow.add_node("testing",             testing_node)
    workflow.add_node("validation",          validation_node)
    workflow.add_node("failure_analyzer",    failure_analyzer_node)
    workflow.add_node("deployment",          deployment_node)

    # ── Wire the entry point ────────────────────────────────────────────────
    # All subsequent transitions are driven by Command(goto=...) from each node.
    # The failure_analyzer → model_building / preprocessing / END paths are
    # handled inside each agent's Command return — no conditional edges needed.
    workflow.add_edge(START, "problem_analyzer")

    # ── Compile graph ────────────────────────────────────────────────────────
    if checkpointer is not None:
        return workflow.compile(checkpointer=checkpointer)
    return workflow.compile()
