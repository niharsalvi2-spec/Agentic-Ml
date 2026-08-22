"""
Canonical AgentEvent schema for the Agentic ML Platform SSE event stream.

Design contract:
  - Every SSE data frame emitted from the backend MUST be one of these AgentEvent subtypes.
  - The frontend renders ONLY from these events — no hardcoded stage sequences.
  - event_type drives frontend routing; agent_id identifies which pipeline node fired.
  - Evidence is structured — never a raw string describing "what the LLM said".
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict
import json


class AgentEventType(str, Enum):
    RUN_STARTED            = "run_started"
    AGENT_STARTED          = "agent_started"
    DECISION_CREATED       = "decision_created"
    TOOL_STARTED           = "tool_started"
    TOOL_COMPLETED         = "tool_completed"
    ARTIFACT_CREATED       = "artifact_created"
    VALIDATION_RESULT      = "validation_result"
    HUMAN_APPROVAL_REQUIRED = "human_approval_required"
    AGENT_COMPLETED        = "agent_completed"
    AGENT_FAILED           = "agent_failed"
    RUN_COMPLETED          = "run_completed"
    RUN_FAILED             = "run_failed"


@dataclass
class AgentEvidence:
    """Structured evidence produced by an agent — never a raw LLM string."""
    rows:             Optional[int]            = None
    columns:          Optional[int]            = None
    features_before:  Optional[int]            = None
    features_after:   Optional[int]            = None
    missing_pct:      Optional[float]          = None
    task_type:        Optional[str]            = None
    target_column:    Optional[str]            = None
    target_inference: Optional[str]            = None   # "explicit" | "last_column"
    candidates:       Optional[List[str]]      = None
    best_model:       Optional[str]            = None
    metrics:          Optional[Dict[str, Any]] = None
    artifact_id:      Optional[str]            = None
    artifact_sha256:  Optional[str]            = None
    warnings:         Optional[List[str]]      = None
    extra:            Optional[Dict[str, Any]] = None


@dataclass
class AgentDecisionRecord:
    """Structured decision record — what the agent decided, why, and with what confidence."""
    agent:         str
    objective:     str
    observations:  str
    decision:      str
    selected_tool: str
    reason:        str
    confidence:    float
    constraints:   List[str]        = field(default_factory=list)
    timestamp:     str              = ""


@dataclass
class AgentEvent:
    """Single event in the SSE stream. Frontend renders from this; never from assumptions."""
    event_type:    AgentEventType
    run_id:        str
    agent_id:      str
    agent_name:    str
    timestamp:     str
    stage_index:   int              = 0
    total_stages:  int              = 10
    is_final:      bool             = False

    # Optional payloads — present only in relevant event types
    message:       Optional[str]    = None
    evidence:      Optional[AgentEvidence] = None
    decision:      Optional[AgentDecisionRecord] = None
    tool_name:     Optional[str]    = None
    duration_ms:   Optional[float]  = None
    artifact_path: Optional[str]    = None
    summary:       Optional[Dict[str, Any]] = None
    risk_score:    Optional[int]    = None
    risk_level:    Optional[str]    = None
    error:         Optional[str]    = None

    # Compatibility aliases
    agent:         Optional[str]    = None
    stage_name:    Optional[str]    = None
    status:        Optional[str]    = None

    def to_sse(self) -> str:
        """Serialize to Server-Sent Event wire format."""
        data = asdict(self)
        # Convert enum to string value
        data["event_type"] = self.event_type.value
        if not data.get("agent"):
            data["agent"] = self.agent_id
        if not data.get("stage_name"):
            data["stage_name"] = self.agent_name
        if not data.get("status"):
            if self.event_type == AgentEventType.RUN_STARTED or self.event_type == AgentEventType.AGENT_STARTED:
                data["status"] = "STARTED"
            elif self.event_type == AgentEventType.AGENT_COMPLETED or self.event_type == AgentEventType.RUN_COMPLETED:
                data["status"] = "COMPLETED"
            elif self.event_type == AgentEventType.AGENT_FAILED or self.event_type == AgentEventType.RUN_FAILED:
                data["status"] = "FAILED"
            else:
                data["status"] = self.event_type.value.upper()
        # Remove None values to keep payload lean
        data = {k: v for k, v in data.items() if v is not None}
        return f"data: {json.dumps(data)}\n\n"


# ── Stage metadata registry ─────────────────────────────────────────────────────
# Single source of truth for stage display names and indices.
# Frontend reads this from events — it does NOT maintain its own hardcoded list.
STAGE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "problem_analyzer":    {"index": 1,  "name": "Problem Analyzer",    "short": "Analyze"},
    "data_collector":      {"index": 2,  "name": "Data Collector",      "short": "Ingest"},
    "preprocessing":       {"index": 3,  "name": "Data Preprocessor",   "short": "Clean"},
    "eda":                 {"index": 4,  "name": "EDA Profiler",        "short": "EDA"},
    "feature_engineering": {"index": 5,  "name": "Feature Engineering", "short": "Engineer"},
    "feature_selection":   {"index": 6,  "name": "Feature Selection",   "short": "Select"},
    "model_building":      {"index": 7,  "name": "Model Building",      "short": "Train"},
    "testing":             {"index": 8,  "name": "Testing QA",          "short": "Test"},
    "validation":          {"index": 9,  "name": "Validation Gate",     "short": "Validate"},
    "failure_analyzer":    {"index": 9,  "name": "Failure Analyzer",    "short": "Analyze"},
    "deployment":          {"index": 10, "name": "Deployment Gate",     "short": "Deploy"},
}


def stage_meta(agent_id: str) -> Dict[str, Any]:
    """Return display metadata for an agent node."""
    return STAGE_REGISTRY.get(agent_id, {"index": 0, "name": agent_id.title(), "short": agent_id})
