"""
Canonical AgentState contract for the Agentic ML Engineering Platform.

Design contract:
  - AgentState carries evidence, outputs, metrics, live data artifacts, and provenance.
  - AgentState does NOT carry routing instructions (NO next_agent field).
  - Routing is exclusively handled by LangGraph Command returns from agent nodes.
  - A completion flag (e.g. data_collected=True) must only be set after the
    deterministic engine operation succeeds and produces verifiable output.
"""
from typing import TypedDict, Annotated, Sequence, Optional, Dict, Any, List
import operator
from langchain_core.messages import BaseMessage


class AgentProvenance(TypedDict):
    """Audit trail entry written by each agent after it completes real work."""
    agent_name: str
    timestamp: str           # ISO-8601 UTC
    operation: str           # Human-readable description of what was done
    result_summary: str      # Short plaintext summary of the result
    artifact_path: Optional[str]  # Absolute path to any artifact produced


class AgentState(TypedDict, total=False):
    """
    The central state contract of the ML Engineer orchestrator graph.

    Split into five semantic layers:
      1. Task context  — the input specification
      2. Data context  — raw and processed dataset metadata + in-memory dataframes
      3. Model context — features, candidates, results, artifacts
      4. Pipeline progression — completion flags (set only on verified success)
      5. Provenance    — per-agent audit trail
    """
    messages: Annotated[Sequence[BaseMessage], operator.add]

    # ── Layer 1: Task Context ────────────────────────────────────────────────
    raw_prompt: str
    current_task: str
    task_type: str          # "classification" | "regression" | "clustering"
    target_column: Optional[str]

    # ── Layer 2: Data Context & Pipeline Data Artifacts ──────────────────────
    dataset_path: str
    dataset_info: Dict[str, Any]    # Schema, shape, dtypes
    data_summary: Dict[str, Any]    # EDA statistics
    raw_df: Any                     # pandas.DataFrame (loaded by data_collector)
    clean_df: Any                   # pandas.DataFrame (cleaned by preprocessing)
    X: Any                          # pandas.DataFrame (feature matrix transformed downstream)
    y: Any                          # pandas.Series / numpy.ndarray (target array)
    preprocessor_obj: Any           # Fitted DeterministicPreprocessor instance

    # ── Layer 3: Model Context ───────────────────────────────────────────────
    selected_features: List[str]
    candidate_models: List[str]
    trained_models: Dict[str, Any]
    best_model_name: Optional[str]
    best_model_metrics: Dict[str, float]

    # ── Artifact Context ─────────────────────────────────────────────────────
    # Path to the versioned artifact directory: artifacts/<model_name>/v<N>/
    artifact_path: Optional[str]

    # ── Layer 4: Pipeline Progression Flags ─────────────────────────────────
    # INVARIANT: a flag is set to True ONLY after the deterministic engine
    # operation succeeds and produces verifiable outputs.
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

    # ── Layer 5: Provenance (Audit Trail) ────────────────────────────────────
    # List of AgentProvenance dicts, one appended per completed agent.
    provenance: Annotated[List[AgentProvenance], operator.add]

    # ── Execution Mode & Status ──────────────────────────────────────────────
    # "live"       — active LLM responses
    # "simulation" — fallback / mock LLM
    execution_mode: str
