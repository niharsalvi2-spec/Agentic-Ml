"""
Unit and integration tests for truthful SSE pipeline streaming.
Verifies event structure, authentic metrics, absence of fake fallbacks, and NO next_agent.
"""
import unittest
import asyncio
import json
from unittest.mock import patch, MagicMock
from langchain_core.messages import AIMessage

from src.agentic_ml.api.routes.pipeline import generate_pipeline_events


class TestSSEStream(unittest.IsolatedAsyncioTestCase):

    @patch("src.agentic_ml.llm.factory.get_llm")
    async def test_full_sse_stream_execution(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="Simulated LLM response for SSE test.")
        mock_get_llm.return_value = mock_llm

        events = []
        async for event_str in generate_pipeline_events("Predict customer churn", target_column="target"):
            events.append(event_str)

        self.assertGreater(len(events), 5)

        # 1. Verify start event
        first_event = events[0]
        self.assertTrue(first_event.startswith("data: "))
        start_payload = json.loads(first_event.replace("data: ", "").strip())
        self.assertEqual(start_payload["agent"], "orchestrator")
        self.assertEqual(start_payload["status"], "STARTED")

        # 2. Check intermediate node events
        agent_names = set()
        for ev in events[1:-1]:
            if ev.strip() == "data: [DONE]":
                continue
            raw_json = ev.replace("data: ", "").strip()
            data = json.loads(raw_json)
            agent_names.add(data.get("agent"))

            # Invariant: NO next_agent anywhere in payloads or snapshots
            self.assertNotIn("next_agent", data)
            if "state_snapshot" in data:
                self.assertNotIn("next_agent", data["state_snapshot"])

            # Verify real stage metadata
            self.assertIn("stage_index", data)
            self.assertIn("stage_name", data)

        # Verify key pipeline stages executed
        self.assertIn("problem_analyzer", agent_names)
        self.assertIn("data_collector", agent_names)
        self.assertIn("model_building", agent_names)
        self.assertIn("validation", agent_names)
        self.assertIn("deployment", agent_names)

        # 3. Check final summary event
        final_event = [e for e in events if '"is_final": true' in e.lower() or '"is_final": True' in e]
        self.assertGreater(len(final_event), 0)
        final_data = json.loads(final_event[-1].replace("data: ", "").strip())
        self.assertTrue(final_data["is_final"])
        self.assertIn("summary", final_data)

        summary = final_data["summary"]
        self.assertIn("selected_model", summary)
        self.assertIn("metrics", summary)
        self.assertIsInstance(summary["metrics"], dict)
        self.assertIn("artifact_path", summary)

    async def test_no_next_agent_in_initial_state(self):
        """Verify contract invariant: next_agent is never introduced."""
        events = []
        async for event_str in generate_pipeline_events("Analyze housing prices"):
            events.append(event_str)
            if len(events) >= 2:
                break
        for ev in events:
            if ev.strip() != "data: [DONE]":
                payload = json.loads(ev.replace("data: ", "").strip())
                self.assertNotIn("next_agent", payload)


if __name__ == "__main__":
    unittest.main()
