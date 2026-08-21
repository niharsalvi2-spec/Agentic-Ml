"""
Chat Streaming API Route for Next.js Chat Interface.
"""

import json
import asyncio
import logging
from typing import AsyncGenerator
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from src.agentic_ml.llm.factory import get_llm
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger("agentic_ml.api.chat")
router = APIRouter()

class ChatRequest(BaseModel):
    prompt: str
    model: str = "Gemini 2.5 Flash"


async def generate_chat_events(user_prompt: str, model_name: str) -> AsyncGenerator[str, None]:
    """
    Streams LLM thoughts and response tokens formatted as Server-Sent Events (SSE).
    """
    thinking_payload = {
        "agent": f"Autonomous ML Assistant ({model_name})",
        "message": f"Analyzing request: '{user_prompt}'\n\n"
    }
    yield f"data: {json.dumps(thinking_payload)}\n\n"
    await asyncio.sleep(0.3)

    try:
        llm = get_llm()
        system_instruction = (
            "You are the Senior Autonomous ML Engineering Assistant. You help data scientists and "
            "engineers formulate machine learning workflows, design features, inspect schemas, and write "
            "clean scikit-learn/PyTorch code according to strict zero-leakage deterministic standards."
        )
        response = llm.invoke([
            SystemMessage(content=system_instruction),
            HumanMessage(content=user_prompt)
        ])
        
        content = response.content if hasattr(response, "content") else str(response)
        
        # Stream content in chunks for responsive UX
        words = content.split(" ")
        chunk_size = 4
        for i in range(0, len(words), chunk_size):
            chunk_text = " ".join(words[i:i+chunk_size]) + " "
            token_payload = {
                "agent": f"Autonomous ML Assistant ({model_name})",
                "message": chunk_text
            }
            yield f"data: {json.dumps(token_payload)}\n\n"
            await asyncio.sleep(0.04)

    except Exception as exc:
        logger.error("Chat streaming error: %s", exc)
        err_payload = {
            "agent": "Autonomous ML Assistant",
            "message": f"\n\n[Error generating response: {str(exc)}]\n"
        }
        yield f"data: {json.dumps(err_payload)}\n\n"

    yield "data: [DONE]\n\n"


@router.post("/chat")
async def chat_endpoint(req: ChatRequest):
    return StreamingResponse(
        generate_chat_events(req.prompt, req.model),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )
