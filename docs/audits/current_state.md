# Current State Snapshot

- **Audit Date**: 2026-08-23
- **Branch**: `main`
- **Latest Commit**: `4414a83` (`docs: Add FINAL_AUDIT.md — 97/97 tests, 100/100 score`)

---

## 1. Environment & Dependencies

- **Python Version**: 3.10+ (Local virtual environment at `./venv/Scripts/python.exe`)
- **Node.js**: Node 20 LTS (Frontend at `./frontend`)
- **Primary Dependencies**:
  - `langgraph >= 0.2.0`, `langchain-core >= 0.3.0`
  - `fastapi >= 0.110.0`, `uvicorn >= 0.30.0`
  - `pydantic >= 2.7.0`
  - `scikit-learn >= 1.4.0`, `pandas >= 2.2.0`, `numpy >= 1.26.0`, `torch >= 2.2.0`
  - `cryptography` (Ed25519 signing)
- **Lockfile Status**: `requirements.lock` and `frontend/package-lock.json` present.

---

## 2. Major Components

1. **Orchestration & State Machine (`src/agentic_ml/orchestration/`, `src/agentic_ml/api/`)**:
   - `build_agentic_graph()`: 10-node LangGraph workflow with dynamic routing, retries, and conditional edges.
   - `RunManager` & `RunRegistry`: SQLite-backed checkpointing and atomic state management for execution streams and HITL.
2. **Deterministic ML Engine (`src/agentic_ml/ml_engine/`)**:
   - Preprocessing & Cleaning: IQR fences, modified Z-score, missingness reports.
   - Encoders: Out-of-fold target encoding, frequency encoding, cardinality categorization.
   - EDA: Freedman-Diaconis binning, multicollinearity detection.
   - Model Registry & Training: Multi-family baseline & candidate training, metric calculation.
   - Evaluation & Risk Scorer: Task-specific metric selection, heuristic model risk assessment.
3. **Artifact Integrity & Security (`src/agentic_ml/security/`, `src/agentic_ml/storage/`)**:
   - Ed25519 asymmetric cryptographic signing & verification.
   - Manifest hashing and provenance tracking (`dataset_hash`, `git_commit`, `random_seed`, etc.).
   - Path traversal protections and dataset boundary enforcement.
4. **API & Real-Time Streaming (`src/agentic_ml/api/`)**:
   - FastAPI server with SSE streaming endpoints (`/api/runs/{run_id}/stream`).
   - HITL approval gate endpoints (`/api/runs/{run_id}/approve`, `/api/runs/{run_id}/reject`).
5. **Frontend UI (`frontend/`)**:
   - Next.js 16 App Router with React Three Fiber 3D canvas and SSE pipeline consumer.

---

## 3. Current Test Commands

- **Python Unit & Integration Test Suite**:
  ```bash
  .\venv\Scripts\python.exe -m pytest tests/ -v
  ```
- **Frontend Strict Quality Gate**:
  ```bash
  cd frontend
  npm run lint
  npx tsc --noEmit
  npm run build
  ```
- **CI Workflow**:
  - Defined in `.github/workflows/ci.yml` covering Python 3.10 lint, import gates, unit tests, security tests, reproducibility tests, integration tests, and frontend strict TypeScript + build.

---

## 4. Known Architectural Boundaries

- **Sandbox Boundary**: Subprocess execution with resource limits and restricted environment variables; isolated container recommended for untrusted code execution.
- **Checkpoint Persistence**: SQLite default checkpointer (`agentic_ml_checkpoints.db`) with fail-closed behavior (no silent fallback to in-memory in production).
- **HITL Gate**: Atomic compare-and-set state transitions (`AWAITING_APPROVAL` -> `APPROVED` / `REJECTED`).
- **Cryptographic Boundary**: Ed25519 signing on artifact bundles and SHA-256 manifests.
