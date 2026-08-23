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
        import pandas as pd
        import numpy as np
        X = pd.DataFrame(np.array([[1, 2], [3, 4], [1, 3]]), columns=["f1", "f2"])
        state: AgentState = {
            "candidate_models": ["RandomForest"],
            "trained_models": {"RandomForest": rf},
            "X": X,
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
            "deployment_decision": "AUTO_APPROVE",
            "trained_models": {"RandomForest": rf},
            "selected_features": ["f1", "f2"],
            "target_column": "target",
            "best_model_metrics": {"accuracy": 0.95},
            "run_id": "run_test_cmd_001",
            "dataset_hash": "7f83b1657ff1fc53b92dc18148a1d65dfc2d4b1fa3d677284addd200126d9069",
            "messages": []
        }
        cmd = deployment_node(state)
        self.assertIsInstance(cmd, Command)
        self.assertEqual(cmd.goto, END)
        self.assertIsNotNone(cmd.update)
        update = cmd.update or {}
        self.assertTrue(update.get("deployment_completed"))
        self.assertIn("artifact_path", update)

    @patch("src.agentic_ml.llm.factory.get_llm")
    @patch("src.agentic_ml.ml_engine.evaluation.validation.ModelEvaluator.evaluate")
    def test_validation_retry_and_exhaustion(self, mock_eval, mock_get_llm):
        mock_get_llm.return_value = self.mock_llm
        # Low score to trigger validation failure
        mock_eval.return_value = ("DummyModel", {"DummyModel": 0.10}, {"DummyModel": 0.01})

        state: AgentState = {
            "task_type": "classification",
            "dataset_path": "",
            "candidate_models": ["DummyModel"],
            "trained_models": {"DummyModel": MagicMock()},
            "selected_features": [],
            "validation_retry_count": 0,
            "messages": []
        }

        # With MAX_RETRIES=2, first failure (retry_count=0) should route to failure_analyzer
        with patch.dict("os.environ", {"MAX_RETRIES": "2"}):
            cmd = validation_node(state)
            self.assertEqual(cmd.goto, "failure_analyzer")
            self.assertEqual(cmd.update.get("validation_retry_count"), 1)

        # With MAX_RETRIES=1 and retry_count=1, failure should exhaust retries and route to END
        state["validation_retry_count"] = 1
        with patch.dict("os.environ", {"MAX_RETRIES": "1"}):
            cmd_exhausted = validation_node(state)
            self.assertEqual(cmd_exhausted.goto, END)
            self.assertFalse(cmd_exhausted.update.get("model_validated"))

        # With MAX_RETRIES=0 (zero retries allowed), even retry_count=0 should route to END immediately
        state["validation_retry_count"] = 0
        with patch.dict("os.environ", {"MAX_RETRIES": "0"}):
            cmd_zero = validation_node(state)
            self.assertEqual(cmd_zero.goto, END)

    @patch("src.agentic_ml.llm.factory.get_llm")
    def test_failure_analyzer_routing_paths(self, mock_get_llm):
        from src.agentic_ml.agents.failure_analyzer.agent import failure_analyzer_node
        mock_get_llm.return_value = self.mock_llm

        # 1. Action retry_with_different_model -> model_building
        state_model: AgentState = {
            "last_failure_analysis": {"remediation_action": "retry_with_different_model", "root_cause": "low accuracy"},
            "messages": []
        }
        cmd1 = failure_analyzer_node(state_model)
        self.assertEqual(cmd1.goto, "model_building")

        # 2. Action retry_preprocessing -> preprocessing
        state_preproc: AgentState = {
            "last_failure_analysis": {"remediation_action": "retry_preprocessing", "root_cause": "severe imbalance"},
            "messages": []
        }
        cmd2 = failure_analyzer_node(state_preproc)
        self.assertEqual(cmd2.goto, "preprocessing")

        # 3. Action flag_and_stop -> END
        state_stop: AgentState = {
            "last_failure_analysis": {"remediation_action": "flag_and_stop", "root_cause": "unrecoverable"},
            "messages": []
        }
        cmd3 = failure_analyzer_node(state_stop)
        self.assertEqual(cmd3.goto, END)


if __name__ == "__main__":
    unittest.main()

