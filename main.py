"""
CLI Entrypoint — Agentic ML Engineering Platform.

Launches the 10-agent LangGraph pipeline from the command line.
All three entrypoints (CLI, FastAPI, MCP) share the same
build_agentic_graph() function from src.agentic_ml.orchestration.graph.
"""
import sys
import logging
from pathlib import Path
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage

# Ensure project root is always on sys.path regardless of invocation method.
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.agentic_ml.orchestration.graph import build_agentic_graph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("agentic_ml.cli")


def build_initial_state(prompt: str, task_type: str = "classification") -> dict:
    """
    Build a clean initial AgentState for the pipeline.

    Args:
        prompt:    Natural language task description.
        task_type: One of "classification", "regression", "clustering".

    Returns:
        dict: Fully initialised AgentState-compatible dictionary.
    """
    return {
        # Messages
        "messages": [HumanMessage(content=prompt)],

        # Layer 1: Task context
        "raw_prompt":     prompt,
        "current_task":   prompt,
        "task_type":      task_type,
        "target_column":  None,

        # Layer 2: Data context
        "dataset_path":  "",
        "dataset_info":  {},
        "data_summary":  {},

        # Layer 3: Model context
        "selected_features":  [],
        "candidate_models":   [],
        "trained_models":     {},
        "best_model_name":    None,
        "best_model_metrics": {},

        # Artifact context
        "artifact_path": None,

        # Layer 4: Pipeline progression flags
        "problem_analyzed":           False,
        "data_collected":             False,
        "data_preprocessed":          False,
        "eda_completed":              False,
        "feature_engineered":         False,
        "feature_selection_completed": False,
        "model_built":                False,
        "model_tested":               False,
        "model_validated":            False,
        "deployment_completed":       False,

        # Layer 5: Provenance (empty list, agents append to it)
        "provenance": [],

        # Execution mode (agents will set to "live" or "simulation")
        "execution_mode": "unknown",
    }


def main():
    prompt = "I want an optimal model to predict Customer Churn."

    print("=" * 60)
    print("  Agentic ML Engineering Platform — CLI")
    print("=" * 60)
    print(f"  Task   : {prompt}")
    print(f"  Started: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60 + "\n")

    app = build_agentic_graph()
    initial_state = build_initial_state(prompt, task_type="classification")

    final_state = None
    for output in app.stream(initial_state):
        for node_name, state_update in output.items():
            logger.info("[%s] COMPLETED", node_name.upper())
            if "messages" in state_update and state_update["messages"]:
                last_msg = state_update["messages"][-1].content
                print(f"  └─ {last_msg[:200]}\n")
            final_state = state_update

    print("\n" + "=" * 60)
    if final_state and final_state.get("artifact_path"):
        print(f"  [+] Artifact: {final_state['artifact_path']}")
    print(f"  [+] Execution mode: {final_state.get('execution_mode', 'unknown') if final_state else 'unknown'}")
    print(f"  [+] Pipeline complete. {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
