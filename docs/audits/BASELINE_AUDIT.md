# Baseline Audit: Agentic-ML Platform

**Date**: 2026-08-23  
**Commit SHA**: `2c631ce3dc067fd87eab1179a4e985a8dd6eadae`  
**Environment**: Python 3.10.11, Node v24.16.0, win32

---

## 1. System Architecture & Components
- **Orchestration**: LangGraph StateGraph connecting 12 nodes (`problem_analyzer`, `data_collector`, `preprocessing`, `eda`, `feature_engineering`, `feature_selection`, `model_building`, `testing`, `validation`, `failure_analyzer`, `deployment_gate`, `deployment`).
- **Execution Manager**: `RunManager` handling initial state creation, checkpointer setup, SSE streaming, and HITL resume flows.
- **Persistence**: SQLite-backed `RunRegistry` for metadata and approval states, LangGraph checkpointers for durable execution state.
- **ML Engine**: Preprocessing, Feature Engineering, Feature Selection, Model Training, Metric Evaluation, and PKL Bundle Generation.
- **Security & Integrity**: Sandbox runner with isolation/limits, Ed25519 signature generator, SHA-256 manifest verifier.
- **Frontend**: Next.js (App Router) + React + TailwindCSS dashboard displaying pipeline progression, logs, charts, and HITL decision cards.

---

## 2. Identified Defects & Severity Matrix

| ID | Component | Severity | Description | Affected Files | Proposed Fix |
|---|---|---|---|---|---|
| **DEF-01** | Orchestration & Checkpointing | **P0** | `build_agentic_graph()` called without `checkpointer` in `stream_run()` and `resume_run()`. Silent fallback to `MemorySaver` disguises lack of persistence. | `src/agentic_ml/api/run_manager.py`, `src/agentic_ml/orchestration/graph.py` | Pass explicit checkpointer to `build_agentic_graph(checkpointer=...)`. Enforce `CHECKPOINT_BACKEND=sqlite` by default and fail explicitly if unavailable. |
| **DEF-02** | HITL Governance & Concurrency | **P0** | Non-atomic compare-and-set in `RunRegistry.resolve_hitl_approval()` allows race conditions in concurrent approval requests. | `src/agentic_ml/storage/run_registry.py` | Implement atomic SQL `UPDATE ... WHERE status = 'AWAITING_APPROVAL'` and assert `cursor.rowcount == 1`. |
| **DEF-03** | Deployment Policy Contradiction | **P0** | Medium risk runs (score 40–69) silently auto-approve despite governance policy dictating `HUMAN_REQUIRED`. | `src/agentic_ml/agents/deployment_gate/agent.py`, `src/agentic_ml/ml_engine/evaluation/risk_scorer.py` | Centralize `DeploymentPolicy` thresholds (LOW <40 Auto, MEDIUM 40–69 HITL, HIGH 70–100 HITL) without silent bypass. |
| **DEF-04** | Fallback / Fabricated Data | **P0** | Hardcoded fallbacks (`metric or 0.0`, `"best_model_name" or "RandomForestClassifier"`, `"unknown"` hashes) can create false success. | `src/agentic_ml/agents/*`, `src/agentic_ml/api/run_manager.py` | Enforce fail-closed architecture. Reject missing upstream evidence. |
| **DEF-05** | Provenance & RunContext | **P1** | Loose dict-based provenance without strict context validation allows incomplete deployment artifacts. | `src/agentic_ml/core/context.py`, `src/agentic_ml/security/manifest.py` | Introduce strongly-typed `RunContext` required for artifact manifest and signature generation. |
| **DEF-06** | Metric Evaluation & Comparison | **P1** | Use of generic `max(metrics.values())` compares incompatible metric scales (e.g. RMSE vs R²). | `src/agentic_ml/api/run_manager.py`, `src/agentic_ml/ml_engine/evaluation/metrics.py` | Use centralized `MetricRegistry` and `PrimaryMetric` aware of task type and direction (minimize vs maximize). |
| **DEF-07** | Event Monotonicity across Resume | **P1** | Sequence numbers hardcoded to 100 on HITL resume rather than monotonically continuing from last sequence. | `src/agentic_ml/api/run_manager.py` | Track and persist monotonic sequence numbers across pauses and resumes. |
| **DEF-08** | Frontend Lifecycle & React Warnings | **P2** | `pipeline/page.tsx` has `setState` in effect, variable access before declaration, and unused parameters. | `frontend/src/app/pipeline/page.tsx` | Refactor React state synchronization, order function definitions, clean unused variables. |
| **DEF-09** | Dependency Lock & Reproducibility | **P2** | Missing exact `requirements.lock` file leads to non-deterministic dependency installations in CI. | `requirements.lock`, `pyproject.toml` | Generate verified `requirements.lock` pinning all direct and transitive dependencies. |

---

## 3. Action Plan
Proceed through Phases 1–35 sequentially, verifying unit and integration tests at each stage.
