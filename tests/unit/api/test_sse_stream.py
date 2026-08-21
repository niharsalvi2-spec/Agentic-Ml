import unittest
import asyncio
import json
from src.agentic_ml.api.routes.pipeline import generate_pipeline_events

class TestSSEStream(unittest.IsolatedAsyncioTestCase):
    async def test_sse_stream_events(self):
        events = []
        async for event_str in generate_pipeline_events("Predict customer churn"):
            events.append(event_str)
            if len(events) >= 3:
                break
                
        self.assertGreater(len(events), 0)
        first_event = events[0]
        self.assertTrue(first_event.startswith("data: "))
        payload = json.loads(first_event.replace("data: ", "").strip())
        self.assertIn("agent", payload)
        self.assertIn("status", payload)

if __name__ == "__main__":
    unittest.main()
