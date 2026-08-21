# Patent-Grade Architecture & Repository Audit Manifest
**Project**: Autonomous Agentic ML Engineering Platform  
**Specification Level**: Industrial Enterprise / Audit-Ready  
**Framework**: LangGraph Multi-Agent Orchestrator, FastAPI Microservices, Next.js 16 UI Studio  

---

## 1. System Structural Overview

```
Agentic ML Engineer/
├── .github/workflows/                 # Continuous Integration & Automated Tests
├── artifacts/                         # Version-Controlled ML Lifecycle Artifacts
│   ├── datasets/                      # Ingested & cached benchmark datasets
│   ├── models/                        # Serialized model weights & transformers
│   ├── plots/                         # Generated diagnostic plots (Matplotlib/Seaborn)
│   ├── reports/                       # Formal Model Cards & Evaluation Audits
│   └── runs/                          # Granular experiment execution traces
├── data/                              # Data Lake Storage
│   ├── raw/                           # Immutable raw inputs
│   ├── interim/                       # Intermediate stage transformed matrices
│   ├── processed/                     # Leakage-safe train/test splits
│   ├── sample/                        # Built-in synthetic/domain benchmarks
│   └── external/                      # External benchmark corpora
├── deterministic_brain/               # Deterministic Agent Skills & Rules
│   ├── Data_Encoding_Agent/           # Leakage-safe categorical encoding skills
│   ├── Dataset_Cleaner_Agent/         # Diagnostic cleaning & outlier fences
│   └── Dataset_Collector_Agent/       # Acquisition skeletons & retry backoff
├── docs/                              # Audit & Patent-Grade Documentation
│   ├── architecture/                  # System, workflow, state, deployment specs
│   ├── decisions/                     # Architecture Decision Records (ADRs 001-005)
│   ├── agents/                        # Complete technical specs for agents 01-10
│   └── experiments/                   # Experiment protocol & evaluation logs
├── frontend/                          # Next.js 16 Studio & Real-Time Colab Interface
│   ├── src/app/                       # App Router (/pipeline, /chat, /login, etc.)
│   ├── src/components/                # Live ColabNotebook, 10-Agent Matrix, etc.
│   └── src/lib/                       # Frontend API clients & utility hooks
├── knowledge/                         # Domain Knowledge Base & Vector Index
│   ├── datasets/                      # Dataset profiles & metadata
│   ├── documents/                     # Research specifications & papers
│   ├── experiment_knowledge/          # Past experiment outcomes & hyperparameter logs
│   └── ml_guidelines/                 # Leakage prevention & production guidelines
├── logs/                              # Execution & Daemon Logs (gitignored)
├── scripts/                           # Production Automation & Operational CLI
│   ├── run_demo.py                    # Multi-service launcher & health checker
│   ├── build_vector_index.py          # Vector embedding builder
│   ├── cleanup_artifacts.py           # Artifact lifecycle management
│   ├── evaluate_pipeline.py           # Automated evaluation runner
│   ├── ingest_knowledge.py            # Knowledge base ingestion
│   └── check_status.ps1               # Infrastructure health monitor
├── src/agentic_ml/                    # Unified Core Engine Source Package
│   ├── agents/                        # 10 Autonomous Pipeline Agent Packages
│   │   ├── problem_analyzer/          # Agent 01: Schema & Metric Formulation
│   │   ├── data_collector/            # Agent 02: Resilient Data Ingestion
│   │   ├── preprocessing/             # Agent 03: Leakage-Safe Cleaning
│   │   ├── eda/                       # Agent 04: Statistical Profiling
│   │   ├── feature_engineering/       # Agent 05: Non-Linear Transformations
│   │   ├── feature_selection/         # Agent 06: Mutual Information Ranking
│   │   ├── model_building/            # Agent 07: Candidate Training (GBM/RF/Linear)
│   │   ├── testing/                   # Agent 08: Schema QA & Invariance Checks
│   │   ├── validation/                # Agent 09: 5-Fold Cross-Validation Gate
│   │   └── deployment/                # Agent 10: Model Packaging & Inference
│   ├── api/                           # FastAPI Microservice Router
│   │   ├── main.py                    # Application factory & CORS middleware
│   │   └── routes/                    # pipeline, chat, prediction, artifacts, health
│   ├── core/                          # Constants, Enums, and Exception definitions
│   ├── llm/                           # LLM Providers (Gemini, Groq, Ollama) & Factory
│   ├── mcp/                           # Model Context Protocol (MCP) Server
│   ├── memory/                        # Short-Term, Long-Term, and Vector Memory
│   ├── ml_engine/                     # Scikit-Learn data, eda, and evaluation pipelines
│   ├── model_engine/                  # Custom Transformer Foundation Model & Config
│   ├── orchestration/                 # LangGraph Multi-Agent StateGraph Workflow
│   ├── rag/                           # Ingestion, Chunking, Embeddings, Reranking
│   ├── reporting/                     # Automated Model Cards & Evaluation Reports
│   ├── services/                      # Domain Services (Artifact, Dataset, ML)
│   ├── state/                         # Global AgentState schema definition
│   ├── storage/                       # Filesystem, RunStore, and Database Client
│   └── tools/                         # Agent Execution Tools
├── tests/                             # Unit, Integration, and E2E Test Suite
├── main.py                            # CLI Platform Entrypoint
├── mcp_server.py                      # MCP Stdio Server Entrypoint
├── pyproject.toml                     # Python packaging configuration
├── README.md                          # Platform Architecture & Quickstart Guide
├── run_demo.ps1                       # Windows PowerShell Demo Launcher
└── run_demo.bat                       # Windows Batch Demo Launcher
```

---

## 2. Agent Subsystem Directory Standard

Each of the 10 agents under `src/agentic_ml/agents/<agent_name>/` follows an identical, audit-compliant internal architecture:
- `agent.py`: LangGraph node function handling state transitions.
- `schemas.py`: Pydantic input/output schemas enforcing deterministic contract validation.
- `prompts/`: Version-controlled system and task prompt templates (`system.txt`, `task.txt`).
- `skills/`: Modular Python routines implementing atomic agent competencies.
- `tools/`: Specialized tools callable by the agent (e.g. schema inspection, metrics computation).
- `services/`: Encapsulated domain service for business logic execution.
- `memory/`: Agent-specific namespace and short-term memory isolation.

---

## 3. Data Flow & Execution Pipeline

1. **Problem Formulation (`problem_analyzer`)**: Natural language prompt parsed into strict ML task contract.
2. **Data Acquisition (`data_collector`)**: Resilient fetching/synthesis with exponential backoff.
3. **Data Cleaning (`preprocessing`)**: Zero-leakage rule (fit transformers on train split only; IQR fences).
4. **Data Profiling (`eda`)**: Skewness/kurtosis calculations and dynamic correlation heatmap generation.
5. **Feature Engineering (`feature_engineering`)**: Interaction terms, log transforms, and ratios.
6. **Feature Selection (`feature_selection`)**: Mutual Information computation and ranking.
7. **Model Training (`model_building`)**: Parallel candidate fit (Gradient Boosting, Random Forest, Linear).
8. **QA Testing (`testing`)**: Input/output schema assertions, invariance checks, and latency benchmarking.
9. **Validation Gate (`validation`)**: 5-Fold Stratified Cross-Validation for Champion selection.
10. **Packaging (`deployment`)**: Final model validation and readiness audit.
