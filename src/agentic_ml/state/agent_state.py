"""
Canonical AgentState contract for the Agentic ML Engineering Platform.

Design contract:
  - AgentState carries evidence, outputs, metrics, artifact references, and provenance.
  - AgentState does NOT carry routing instructions (NO next_agent field).
  - Routing is exclusively handled by LangGraph Command returns from agent nodes.
  - A completion flag (e.g. data_collected=True) must only be set after the
    deterministic engine operation succeeds and produces verifiable output.
  - Raw DataFrames (raw_df, clean_df, X, y) are retained for backward compatibility
    but agents should prefer artifact_refs for inter-agent data transfer.
  - run_id, random_seed, python_version are immutable after creation by RunManager.
"""
from typing import TypedDict, Annotated, Sequence, Optional, Dict, Any, List
import operator
from langchain_core.messages import BaseMessage


# ── Provenance ────────────────────────────────────────────────────────────────

class AgentProvenance(TypedDict):
    """Audit trail entry written by each agent after it completes real work."""
    agent_name:     str
    timestamp:      str             # ISO-8601 UTC
    operation:      str             # Human-readable description of what was done
    result_summary: str             # Short plaintext summary of the result
    artifact_path:  Optional[str]   # Absolute path to any artifact produced


# ── Evidence ─────────────────────────────────────────────────────────────────

class AgentEvidenceEntry(TypedDict, total=False):
    """Structured evidence record produced by each agent — NOT a raw LLM string."""
    agent_name:     str
    decision:       str
    selected_tool:  str
    reason:         str
    confidence:     float
    artifacts:      List[str]
    metrics:        Dict[str, Any]
    warnings:       List[str]
    timestamp:      str


# ── Artifact References ───────────────────────────────────────────────────────

class ArtifactRefs(TypedDict, total=False):
    """
    Artifact filesystem references passed between agents.
    Agents consume these refs to load data — they do NOT re-preprocess from scratch.
    """
    dataset_manifest_ref:  Optional[str]   # path to dataset_manifest.json
    preprocessing_ref:     Optional[str]   # path to saved DeterministicPreprocessor
    feature_ref:           Optional[str]   # path to features.parquet
    model_ref:             Optional[str]   # path to model artifact dir
    evaluation_ref:        Optional[str]   # path to evaluation metrics JSON
    bundle_ref:            Optional[str]   # path to signed artifact bundle dir


# ── Execution Error ───────────────────────────────────────────────────────────

class ExecutionError(TypedDict):
    agent_name: str
    error_type: str
    message:    str
    timestamp:  str
    fatal:      bool


# ── Main State ────────────────────────────────────────────────────────────────

class AgentState(TypedDict, total=False):
    """
    The central state contract of the ML Engineer orchestrator graph.

    Semantic layers:
      1. Run context       — immutable metadata set by RunManager on start
      2. Task context      — problem specification
      3. Data context      — dataset metadata and artifact references
      4. Model context     — trained models, features, metrics
      5. Artifact context  — versioned artifact bundle references
      6. Pipeline flags    — completion flags (set ONLY on verified success)
      7. Evidence          — per-agent structured evidence records
      8. Provenance        — per-agent audit trail (legacy, keep for compatibility)
      9. Execution errors  — error records from failed operations
    """
    messages: Annotated[Sequence[BaseMessage], operator.add]

    # ── Layer 1: Run Context (set by RunManager, immutable after start) ──────
    run_id:          str    # e.g. "run_20260822_182901_a83f"
    random_seed:     int    # fixed for reproducibility (default: 42)
    python_version:  str    # sys.version at run start
    started_at:      str    # ISO-8601 UTC

    # ── Layer 2: Task Context ────────────────────────────────────────────────
    raw_prompt:    str
    current_task:  str
    task_type:     str             # "classification" | "regression" | "clustering"
    target_column: Optional[str]   # None means clustering or user did not specify

    # target_inference_method: set if target_column was inferred, not specified by user
    target_inference_method: Optional[str]   # "explicit" | "last_column" | "none"

    # ── Layer 3: Data Context ────────────────────────────────────────────────
    dataset_path:    str
    dataset_hash:    str             # sha256 of raw dataset file
    dataset_info:    Dict[str, Any]  # Schema, shape, dtypes
    data_summary:    Dict[str, Any]  # EDA statistics

    # In-memory dataframes (preferred to use artifact_refs in newer agents)
    raw_df:          Any             # pandas.DataFrame (loaded by data_collector)
    clean_df:        Any             # pandas.DataFrame (cleaned by preprocessing)
    X:               Any             # pandas.DataFrame (feature matrix)
    y:               Any             # pandas.Series / numpy.ndarray (target)
    preprocessor_obj: Any            # Fitted DeterministicPreprocessor instance

    # ── Layer 4: Model Context ───────────────────────────────────────────────
    selected_features:   List[str]
    candidate_models:    List[str]
    trained_models:      Dict[str, Any]
    best_model_name:     Optional[str]
    best_model_metrics:  Dict[str, float]

    # Retry counter for validation → model_building feedback loop
    validation_retry_count:  int
    last_failure_analysis:   Optional[Dict[str, Any]]  # FailureAnalysis dict

    # Risk scoring (set by deployment agent before HITL interrupt)
    risk_score:  Optional[int]    # 0-100
    risk_level:  Optional[str]    # "LOW" | "MEDIUM" | "HIGH"

    # HITL approval decision (set by deployment_gate before deployment runs)
    # Values: "AUTO_APPROVE" | "HUMAN_APPROVED" | "REJECTED" | None
    deployment_decision: Optional[str]

    # ── Layer 5: Artifact References ─────────────────────────────────────────
    # Agents consume artifact_refs to load data; they do NOT re-preprocess
    artifact_refs:   ArtifactRefs

    # Legacy: direct path to the versioned bundle dir (kept for API compatibility)
    artifact_path:   Optional[str]

    # ── Layer 6: Pipeline Progression Flags ─────────────────────────────────
    # INVARIANT: a flag is set to True ONLY after the deterministic engine
    # operation succeeds and produces verifiable outputs.
    problem_analyzed:           bool
    data_collected:             bool
    data_preprocessed:          bool
    eda_completed:              bool
    feature_engineered:         bool
    feature_selection_completed: bool
    model_built:                bool
    model_tested:               bool
    model_validated:            bool
    deployment_completed:       bool

    # ── Layer 7: Structured Evidence ─────────────────────────────────────────
    # List of AgentEvidenceEntry dicts, one appended per completed agent.
    # Reducers: operator.add means new entries are appended (not overwritten).
    evidence: Annotated[List[AgentEvidenceEntry], operator.add]

    # ── Layer 8: Provenance (Audit Trail) ────────────────────────────────────
    # List of AgentProvenance dicts, one appended per completed agent.
    provenance: Annotated[List[AgentProvenance], operator.add]

    # ── Layer 9: Execution Errors ────────────────────────────────────────────
    errors: Annotated[List[ExecutionError], operator.add]

    # ── Execution Mode ────────────────────────────────────────────────────────
    # "live"       — active LLM responses
    # "simulation" — fallback / mock LLM
    execution_mode: str
