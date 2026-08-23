import unittest
import sys
from pathlib import Path
from langchain_core.messages import HumanMessage

# Add root directory to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

from src.agentic_ml.orchestration.graph import build_agentic_graph

class TestFullPipeline(unittest.TestCase):
    def test_full_pipeline_execution(self):
        app = build_agentic_graph()
        initial_state = {
            "run_id": "run_test_e2e_001",
            "random_seed": 42,
            "messages": [HumanMessage(content="Test customer churn classification task")],
            "raw_prompt": "Test customer churn classification task",
            "current_task": "Test customer churn classification task",
            "task_type": "classification",
            "target_column": None,
            "dataset_path": "",
            "dataset_info": {},
            "data_summary": {},
            "selected_features": [],
            "candidate_models": [],
            "trained_models": {},
            "best_model_name": None,
            "best_model_metrics": {},
            "artifact_path": None,
            "problem_analyzed": False,
            "data_collected": False,
            "data_preprocessed": False,
            "eda_completed": False,
            "feature_engineered": False,
            "feature_selection_completed": False,
            "model_built": False,
            "model_tested": False,
            "model_validated": False,
            "deployment_completed": False,
        }
        
        final_output = None
        for output in app.stream(initial_state):
            final_output = output
            
        self.assertIsNotNone(final_output)
        self.assertIn("deployment", final_output)
        self.assertTrue(final_output["deployment"]["deployment_completed"])
        self.assertIsNotNone(final_output["deployment"]["artifact_path"])

if __name__ == "__main__":
    unittest.main()
