"""
Unit tests verifying that each agent node returns a valid LangGraph Command
with explicit goto routing and correct state updates.
"""
import unittest
from unittest.mock import patch, MagicMock
from langchain_core.messages import AIMessage
from langgraph.types import Command
from langgraph.graph import END

from src.agentic_ml.state.agent_state import AgentState
from src.agentic_ml.agents.problem_analyzer.agent import problem_analyzer_node
from src.agentic_ml.agents.data_collector.agent import data_collector_node
from src.agentic_ml.agents.preprocessing.agent import preprocessing_node
from src.agentic_ml.agents.eda.agent import eda_node
from src.agentic_ml.agents.feature_engineering.agent import feature_engineering_node
from src.agentic_ml.agents.feature_selection.agent import feature_selection_node
from src.agentic_ml.agents.model_building.agent import model_building_node
from src.agentic_ml.agents.testing.agent import testing_node
from src.agentic_ml.agents.validation.agent import validation_node
from src.agentic_ml.agents.deployment.agent import deployment_node


class TestAgentCommands(unittest.TestCase):

    def setUp(self):
        self.mock_llm = MagicMock()
        self.mock_llm.invoke.return_value = AIMessage(content="Verified LLM output.")

    @patch("src.agentic_ml.llm.factory.get_llm")
    def test_problem_analyzer_command(self, mock_get_llm):
        mock_get_llm.return_value = self.mock_llm
        state: AgentState = {"raw_prompt": "Predict customer churn", "task_type": "", "messages": []}
        cmd = problem_analyzer_node(state)
        self.assertIsInstance(cmd, Command)
        self.assertEqual(cmd.goto, "data_collector")
        self.assertIsNotNone(cmd.update)
        update = cmd.update or {}
        self.assertTrue(update.get("problem_analyzed"))
        self.assertEqual(update.get("task_type"), "classification")

    @patch("src.agentic_ml.llm.factory.get_llm")
    def test_data_collector_command(self, mock_get_llm):
        mock_get_llm.return_value = self.mock_llm
        state: AgentState = {"task_type": "classification", "dataset_path": "", "messages": []}
        cmd = data_collector_node(state)
        self.assertIsInstance(cmd, Command)
        self.assertEqual(cmd.goto, "preprocessing")
        self.assertIsNotNone(cmd.update)
        update = cmd.update or {}
        self.assertTrue(update.get("data_collected"))
        self.assertIn("target_column", update)

    @patch("src.agentic_ml.llm.factory.get_llm")
    def test_preprocessing_command(self, mock_get_llm):
        mock_get_llm.return_value = self.mock_llm
        state: AgentState = {"task_type": "classification", "dataset_path": "", "messages": []}
        cmd = preprocessing_node(state)
        self.assertIsInstance(cmd, Command)
        self.assertEqual(cmd.goto, "eda")
        self.assertIsNotNone(cmd.update)
        update = cmd.update or {}
        self.assertTrue(update.get("data_preprocessed"))

    @patch("src.agentic_ml.llm.factory.get_llm")
    def test_eda_command(self, mock_get_llm):
        mock_get_llm.return_value = self.mock_llm
        state: AgentState = {"task_type": "classification", "dataset_path": "", "messages": []}
        cmd = eda_node(state)
        self.assertIsInstance(cmd, Command)
        self.assertEqual(cmd.goto, "feature_engineering")
        self.assertIsNotNone(cmd.update)
        update = cmd.update or {}
        self.assertTrue(update.get("eda_completed"))

    @patch("src.agentic_ml.llm.factory.get_llm")
    def test_feature_engineering_command(self, mock_get_llm):
        mock_get_llm.return_value = self.mock_llm
        state: AgentState = {"task_type": "classification", "dataset_path": "", "messages": []}
        cmd = feature_engineering_node(state)
        self.assertIsInstance(cmd, Command)
        self.assertEqual(cmd.goto, "feature_selection")
        self.assertIsNotNone(cmd.update)
        update = cmd.update or {}
        self.assertTrue(update.get("feature_engineered"))

    @patch("src.agentic_ml.llm.factory.get_llm")
    def test_feature_selection_command(self, mock_get_llm):
        mock_get_llm.return_value = self.mock_llm
        state: AgentState = {"task_type": "classification", "dataset_path": "", "messages": []}
        cmd = feature_selection_node(state)
        self.assertIsInstance(cmd, Command)
        self.assertEqual(cmd.goto, "model_building")
        self.assertIsNotNone(cmd.update)
        update = cmd.update or {}
        self.assertTrue(update.get("feature_selection_completed"))

    @patch("src.agentic_ml.llm.factory.get_llm")
    def test_model_building_command(self, mock_get_llm):
        mock_get_llm.return_value = self.mock_llm
        state: AgentState = {"task_type": "classification", "dataset_path": "", "selected_features": [], "messages": []}
        cmd = model_building_node(state)
        self.assertIsInstance(cmd, Command)
        self.assertEqual(cmd.goto, "testing")
        self.assertIsNotNone(cmd.update)
        update = cmd.update or {}
        self.assertTrue(update.get("model_built"))
        self.assertIn("trained_models", update)

    @patch("src.agentic_ml.llm.factory.get_llm")
    def test_testing_command(self, mock_get_llm):
        mock_get_llm.return_value = self.mock_llm
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier()
        rf.fit([[1, 2], [3, 4]], [0, 1])
        state: AgentState = {
            "candidate_models": ["RandomForest"],
            "trained_models": {"RandomForest": rf},
            "messages": []
        }
        cmd = testing_node(state)
        self.assertIsInstance(cmd, Command)
        self.assertEqual(cmd.goto, "validation")
        self.assertIsNotNone(cmd.update)
        update = cmd.update or {}
        self.assertTrue(update.get("model_tested"))

    @patch("src.agentic_ml.llm.factory.get_llm")
    def test_validation_command(self, mock_get_llm):
        mock_get_llm.return_value = self.mock_llm
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier()
        rf.fit([[1, 2], [3, 4]], [0, 1])
        state: AgentState = {
            "task_type": "classification",
            "dataset_path": "",
            "candidate_models": ["RandomForest"],
            "trained_models": {"RandomForest": rf},
            "selected_features": [],
            "messages": []
        }
        cmd = validation_node(state)
        self.assertIsInstance(cmd, Command)
        self.assertEqual(cmd.goto, "deployment_gate")
        self.assertIsNotNone(cmd.update)
        update = cmd.update or {}
        self.assertTrue(update.get("model_validated"))
        self.assertIn("best_model_name", update)

    @patch("src.agentic_ml.llm.factory.get_llm")
    def test_deployment_command(self, mock_get_llm):
        mock_get_llm.return_value = self.mock_llm
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier()
        rf.fit([[1, 2], [3, 4]], [0, 1])
        state: AgentState = {
            "task_type": "classification",
            "best_model_name": "RandomForest",
            "trained_models": {"RandomForest": rf},
            "selected_features": ["f1", "f2"],
            "target_column": "target",
            "best_model_metrics": {"accuracy": 0.95},
            "messages": []
        }
        cmd = deployment_node(state)
        self.assertIsInstance(cmd, Command)
        self.assertEqual(cmd.goto, END)
        self.assertIsNotNone(cmd.update)
        update = cmd.update or {}
        self.assertTrue(update.get("deployment_completed"))
        self.assertIn("artifact_path", update)


if __name__ == "__main__":
    unittest.main()
