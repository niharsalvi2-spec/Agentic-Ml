from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="3D Motion Website API",
    description="Backend API for the 3D Motion-based Full-Stack Application",
    version="1.0.0",
)

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], # Next.js default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Welcome to the 3D Motion Website API"}

@app.get("/health")
async def health_check():
    return {"status": "ok"}

from pydantic import BaseModel
import asyncio
from fastapi.responses import StreamingResponse
import json

class ChatRequest(BaseModel):
    prompt: str
    model: str = "Gemini 3.5 Flash"

from genix_service import genix_app

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    # Hook the actual PyTorch Transformer directly to the Next.js UI!
    return StreamingResponse(genix_app.stream_inference(request.prompt, request.model), media_type="text/event-stream")
