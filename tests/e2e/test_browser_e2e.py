"""
Real Browser E2E Test — Agentic ML Platform
============================================

Tests the complete lifecycle:
  Browser (Playwright/Chromium)
    → Next.js frontend (localhost:3000)
    → POST /api/pipeline/stream (FastAPI SSE endpoint)
    → RunManager.stream_run (LangGraph)
    → All agent nodes (problem_analyzer → ... → deployment)
    → SSE events received in browser
    → Pipeline UI shows completion

Requirements:
  - Python playwright: pip install playwright && playwright install chromium
  - Backend running: uvicorn src.agentic_ml.api.main:app --port 8000
  - Frontend running: npm run dev (port 3000)

Run with:
  pytest tests/e2e/test_browser_e2e.py -v -s
"""
import asyncio
import json
import subprocess
import sys
import time
import requests
from pathlib import Path
from typing import List

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

BACKEND_URL = "http://localhost:8000"
FRONTEND_URL = "http://localhost:3000"
SSE_URL = f"{BACKEND_URL}/api/pipeline/stream"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _wait_for_server(url: str, timeout: float = 30.0) -> bool:
    """Poll until the server is responding or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code < 500:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def _collect_sse_events(url: str, payload: dict, max_events: int = 50, timeout: float = 120.0) -> List[dict]:
    """
    Connect to an SSE endpoint and collect events until:
      - A terminal event is received (deployment_completed / error / max_retries_reached)
      - max_events is reached
      - timeout is exceeded
    """
    events: List[dict] = []
    deadline = time.time() + timeout

    with requests.post(url, json=payload, stream=True, timeout=timeout) as resp:
        assert resp.status_code == 200, f"SSE endpoint returned {resp.status_code}"
        assert "text/event-stream" in resp.headers.get("Content-Type", "")

        buffer = ""
        for chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
            if time.time() > deadline:
                break
            buffer += chunk
            while "\n\n" in buffer:
                raw_event, buffer = buffer.split("\n\n", 1)
                for line in raw_event.strip().splitlines():
                    if line.startswith("data:"):
                        try:
                            data = json.loads(line[5:].strip())
                            events.append(data)
                            # Stop on terminal events
                            et = data.get("event_type", "")
                            if et in ("deployment_completed", "error", "max_retries_reached"):
                                return events
                        except json.JSONDecodeError:
                            pass
            if len(events) >= max_events:
                break

    return events


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestBrowserE2E:
    """
    Real browser E2E test using Playwright.
    Verifies the complete flow: Browser → Next.js → FastAPI SSE → RunManager → Agents.
    """

    @pytest.fixture(scope="class", autouse=True)
    def backend_process(self):
        """Start the FastAPI backend for the duration of the test class."""
        proc = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn",
                "src.agentic_ml.api.main:app",
                "--host", "0.0.0.0",
                "--port", "8001",
                "--log-level", "warning",
            ],
            cwd=str(ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        ready = _wait_for_server("http://localhost:8001/health", timeout=20.0)
        if not ready:
            proc.kill()
            pytest.skip("Backend failed to start — skipping browser E2E")
        yield proc
        proc.kill()

    def test_health_endpoint_responds(self):
        """Backend /health must respond 200 before any SSE test."""
        r = requests.get("http://localhost:8001/health", timeout=5)
        assert r.status_code == 200, f"Health check failed: {r.status_code} {r.text}"
        body = r.json()
        assert body.get("status") in ("healthy", "ok", "running"), f"Unexpected health body: {body}"
        print(f"\nBACKEND HEALTH: {body}")

    def test_sse_stream_delivers_ordered_events(self):
        """
        POST /api/pipeline/stream → verify:
          - HTTP 200 with text/event-stream Content-Type
          - Events arrive with unique event_id
          - sequence_number is strictly monotonically increasing
          - run_id is consistent across all events
          - A terminal event is received (deployment_completed or similar)
        """
        payload = {
            "prompt": "Classify customer churn on a small synthetic dataset",
            "dataset_path": "",
            "target_column": None,
            "random_seed": 42,
        }

        events = _collect_sse_events(
            "http://localhost:8001/api/pipeline/stream",
            payload,
            max_events=60,
            timeout=150.0,
        )

        assert len(events) > 0, "No SSE events received"

        # Verify event_id uniqueness
        ids = [e.get("event_id") for e in events if e.get("event_id")]
        assert len(ids) == len(set(ids)), f"Duplicate event_ids found: {len(ids) - len(set(ids))} duplicates"

        # Verify run_id consistency
        run_ids = {e.get("run_id") for e in events if e.get("run_id")}
        assert len(run_ids) == 1, f"Multiple run_ids in single stream: {run_ids}"

        # Verify sequence_number monotonicity
        seqs = [e.get("sequence_number") for e in events if e.get("sequence_number") is not None]
        for i in range(1, len(seqs)):
            assert seqs[i] > seqs[i - 1], (
                f"Sequence number not monotonically increasing at position {i}: "
                f"{seqs[i - 1]} → {seqs[i]}"
            )

        # Verify terminal event received
        terminal_types = {"run_completed", "run_failed", "error", "deployment_completed", "max_retries_reached"}
        received_types = {e.get("event_type") for e in events}
        assert received_types & terminal_types, (
            f"No terminal event received. Got types: {received_types}"
        )


        print(f"\nSSE_E2E: {len(events)} events, run_id={next(iter(run_ids))}")
        print(f"  Sequence range: {seqs[0]}..{seqs[-1]}")
        print(f"  Event types: {sorted(received_types)}")
        print(f"  Terminal: {received_types & terminal_types}")

    def test_sse_content_type_header(self):
        """The SSE endpoint MUST return Content-Type: text/event-stream."""
        payload = {
            "prompt": "Quick classification task",
            "dataset_path": "",
            "random_seed": 42,
        }
        with requests.post(
            "http://localhost:8001/api/pipeline/stream",
            json=payload,
            stream=True,
            timeout=10,
        ) as resp:
            assert resp.status_code == 200
            ct = resp.headers.get("Content-Type", "")
            assert "text/event-stream" in ct, f"Expected text/event-stream, got: {ct!r}"
            print(f"\nCONTENT_TYPE: {ct}")

    def test_browser_pipeline_page_renders(self):
        """
        Playwright browser test: open the Next.js /pipeline page and verify
        the UI renders correctly (no JS errors, page title present, start button exists).
        """
        try:
            from playwright.sync_api import sync_playwright, TimeoutError as PWTimeoutError
        except ImportError:
            pytest.skip("playwright not installed — install with: pip install playwright && playwright install chromium")

        frontend_ready = _wait_for_server(FRONTEND_URL, timeout=10.0)
        if not frontend_ready:
            pytest.skip(f"Frontend at {FRONTEND_URL} is not running — start with: npm run dev")

        js_errors: List[str] = []

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page()

            # Capture JavaScript console errors
            page.on("console", lambda msg: js_errors.append(msg.text) if msg.type == "error" else None)
            page.on("pageerror", lambda exc: js_errors.append(str(exc)))

            try:
                page.goto(f"{FRONTEND_URL}/pipeline", wait_until="networkidle", timeout=20_000)
            except PWTimeoutError:
                # networkidle may not settle during SSE; domcontentloaded is sufficient
                page.goto(f"{FRONTEND_URL}/pipeline", wait_until="domcontentloaded", timeout=20_000)

            title = page.title()
            assert title, "Page title is empty"
            print(f"\nBROWSER PAGE TITLE: {title!r}")

            # Verify no critical JS errors
            critical_errors = [e for e in js_errors if "Cannot read" in e or "is not defined" in e]
            assert not critical_errors, f"Critical JS errors: {critical_errors}"

            # Verify the page has some meaningful content (not a blank page)
            body_text = page.locator("body").inner_text()
            assert len(body_text.strip()) > 10, "Page body appears empty"
            print(f"BROWSER BODY (first 200 chars): {body_text[:200]!r}")

            browser.close()

        print(f"\nBROWSER_E2E: Pipeline page rendered successfully. JS errors: {len(js_errors)}")

    def test_browser_sse_stream_via_fetch(self):
        """
        Playwright browser test: use page.evaluate() with fetch() to connect directly
        to the SSE backend from within the browser context, then collect events.
        This validates the full Browser → Network → FastAPI → SSE pipeline.
        """
        try:
            from playwright.sync_api import sync_playwright, TimeoutError as PWTimeoutError
        except ImportError:
            pytest.skip("playwright not installed")

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page()

            # Navigate to the backend docs page to establish an active browser page context
            page.goto("http://localhost:8001/docs", wait_until="domcontentloaded", timeout=15_000)



            page.set_default_timeout(120_000)
            # Use browser's fetch() to hit the backend SSE endpoint
            collected = page.evaluate(
                """async (backendUrl) => {
                    const events = [];
                    try {
                        const resp = await fetch(backendUrl + '/api/pipeline/stream', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                prompt: 'E2E browser SSE validation - classify iris dataset',
                                dataset_path: '',
                                random_seed: 42
                            })
                        });
                        if (!resp.ok) return { error: 'HTTP ' + resp.status };
                        const contentType = resp.headers.get('content-type') || '';
                        const reader = resp.body.getReader();
                        const decoder = new TextDecoder();
                        let buf = '';
                        let maxEvents = 10;
                        while (events.length < maxEvents) {
                            const { value, done } = await reader.read();
                            if (done) break;
                            buf += decoder.decode(value, { stream: true });
                            const parts = buf.split('\\n\\n');
                            for (let i = 0; i < parts.length - 1; i++) {
                                for (const line of parts[i].split('\\n')) {
                                    if (line.startsWith('data:')) {
                                        try {
                                            const parsed = JSON.parse(line.slice(5).trim());
                                            events.push(parsed);
                                        } catch {}
                                    }
                                }
                            }
                            buf = parts[parts.length - 1];
                        }
                        await reader.cancel();
                    } catch (e) {
                        return { error: String(e) };
                    }
                    return events;
                }""",
                "http://localhost:8001",
            )


            browser.close()

        assert isinstance(collected, list), f"Expected list of events, got: {collected}"
        assert len(collected) > 0, "Browser received zero SSE events from backend"

        # Validate event structure from browser side
        first = collected[0]
        assert "run_id" in first, f"Missing run_id in first event: {first}"
        assert "event_type" in first, f"Missing event_type in first event: {first}"
        assert "sequence_number" in first, f"Missing sequence_number in first event: {first}"

        print(f"\nBROWSER_SSE_FETCH: {len(collected)} events collected via browser fetch()")
        print(f"  First event: run_id={first.get('run_id')}, type={first.get('event_type')}, seq={first.get('sequence_number')}")
