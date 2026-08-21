# Agent Architecture

Every agent follows a standard contract:
- **Brain/Controller (`agent.py`)**: Evaluates incoming state and orchestrates skills.
- **Skills (`skills/`)**: Modular, reusable agent-level capabilities.
- **Tools (`tools/`)**: Concrete, deterministic executable functions.
- **Services (`services/`)**: Multi-step domain coordinators.
- **Memory (`memory/`)**: Agent-scoped execution memory.
- **Prompts (`prompts/`)**: Structured system and task prompts.
