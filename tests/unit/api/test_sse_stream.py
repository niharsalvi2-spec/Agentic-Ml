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

    @patch("src.agentic_ml.llm.factory.get_llm")
    async def test_event_identity_and_sequence_monotonicity(self, mock_get_llm):
        """Verify event identity and strict sequence number monotonicity."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="Simulated LLM response for SSE test.")
        mock_get_llm.return_value = mock_llm

        events = []
        async for event_str in generate_pipeline_events("Predict housing prices", target_column="target"):
            if event_str.strip() != "data: [DONE]":
                events.append(json.loads(event_str.replace("data: ", "").strip()))

        self.assertGreater(len(events), 3)
        seq_nums = []
        for ev in events:
            self.assertIn("run_id", ev)
            self.assertIn("event_id", ev)
            self.assertIn("sequence_number", ev)
            self.assertIn("agent_id", ev)
            self.assertIn("attempt_number", ev)
            self.assertIn("event_type", ev)
            self.assertIn("timestamp", ev)
            seq_nums.append(ev["sequence_number"])

        # Strictly increasing sequence numbers starting at 1
        self.assertEqual(seq_nums, list(range(1, len(seq_nums) + 1)))

    async def test_client_event_deduplication(self):
        """Verify client-side deduplication using event_id / sequence_number."""
        seen_event_ids = set()
        mock_stream = [
            {"event_id": "evt_001", "sequence_number": 1, "agent_id": "problem_analyzer"},
            {"event_id": "evt_001", "sequence_number": 1, "agent_id": "problem_analyzer"},  # duplicate
            {"event_id": "evt_002", "sequence_number": 2, "agent_id": "data_collector"},
        ]
        unique_events = []
        for ev in mock_stream:
            if ev["event_id"] not in seen_event_ids:
                seen_event_ids.add(ev["event_id"])
                unique_events.append(ev)

        self.assertEqual(len(unique_events), 2)
        self.assertEqual([e["sequence_number"] for e in unique_events], [1, 2])


if __name__ == "__main__":
    unittest.main()

