"""
Integration test for LangGraph Agentic ML Workflow.
Tests the orchestration of nodes using deterministic engines.
"""

import unittest
from unittest.mock import patch, MagicMock
from langchain_core.messages import AIMessage

from src.agentic_ml.orchestration.graph import build_agentic_graph
from src.agentic_ml.state.agent_state import AgentState


class TestAgenticWorkflowIntegration(unittest.TestCase):

    @patch("src.agentic_ml.llm.factory.get_llm")
    def test_full_pipeline_execution(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="Node execution verified.")
        mock_get_llm.return_value = mock_llm

        graph = build_agentic_graph()
        initial_state: AgentState = {
            "messages": [],
            "current_task": "Build a churn classification model.",
            "task_type": "classification",
            "dataset_path": "",
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
            "target_column": "",
            "best_model_name": "",
            "artifact_path": ""
        }

        final_state = graph.invoke(initial_state)
        self.assertTrue(final_state.get("deployment_completed"))
        self.assertTrue(final_state.get("artifact_path"))
        self.assertEqual(final_state.get("task_type"), "classification")


if __name__ == "__main__":
    unittest.main()
