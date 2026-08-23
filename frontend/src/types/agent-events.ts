/**
 * Canonical AgentEvent TypeScript schema matching the backend AgentEvent contract.
 * Frontend renders exclusively from these typed events.
 */

export type AgentEventType =
  | "run_started"
  | "agent_started"
  | "decision_created"
  | "tool_started"
  | "tool_completed"
  | "artifact_created"
  | "validation_result"
  | "human_approval_required"
  | "agent_completed"
  | "agent_failed"
  | "run_completed"
  | "run_failed";

export interface AgentEvidence {
  rows?: number;
  columns?: number;
  features_before?: number;
  features_after?: number;
  missing_pct?: number;
  task_type?: string;
  target_column?: string;
  target_inference?: "explicit" | "common_name" | "last_column" | "deferred" | "none";
  candidates?: string[];
  best_model?: string;
  metrics?: Record<string, number>;
  artifact_id?: string;
  artifact_sha256?: string;
  warnings?: string[];
  extra?: Record<string, unknown>;
}

export interface AgentDecisionRecord {
  agent: string;
  objective: string;
  observations: string;
  decision: string;
  selected_tool: string;
  reason: string;
  confidence: number;
  constraints: string[];
  timestamp: string;
}

export interface AgentEvent {
  event_type: AgentEventType;
  event_id?: string;
  sequence_number?: number;
  run_id: string;
  agent_id: string;
  agent_name: string;
  timestamp: string;
  stage_index: number;
  total_stages: number;
  is_final?: boolean;
  message?: string;
  evidence?: AgentEvidence;
  decision?: AgentDecisionRecord;
  tool_name?: string;
  duration_ms?: number;
  artifact_path?: string;
  summary?: Record<string, unknown>;
  risk_score?: number;
  risk_level?: string;
  error?: string;
  agent?: string;
  stage_name?: string;
  status?: string;
}

export interface AgentRuntimeState {
  id: string;
  name: string;
  status: "idle" | "running" | "completed" | "failed";
  stageIndex: number;
  message?: string;
  timestamp?: string;
  lastEvent?: AgentEvent;
}
