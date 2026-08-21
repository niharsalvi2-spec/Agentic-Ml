# System Architecture

The Agentic ML Engineering platform is structured as an autonomous multi-agent state machine operating over LangGraph.
Decoupled into:
1. Client Layer (Next.js 16 UI / MCP Protocol)
2. Orchestration & Agent State Machine (LangGraph)
3. Specialized Agent Controllers (Agent -> Skill -> Service -> Tool)
4. Deterministic ML Engine (Pure scikit-learn / pandas / numpy)
5. Artifact & Model Store (Production model.pkl bundles)
