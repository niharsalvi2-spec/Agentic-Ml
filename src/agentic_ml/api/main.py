"""
FastAPI Microservice Application Factory for Agentic ML Engineering Platform.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.agentic_ml.api.routes import pipeline, chat, prediction, artifacts, datasets, health

app = FastAPI(
    title="Agentic ML Engineering Platform API",
    description="Autonomous Multi-Agent Machine Learning Workflow Engine",
    version="1.0.0",
)

ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

# CORS configuration for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# Mount all domain microservice routers
app.include_router(health.router, tags=["Health"])
app.include_router(pipeline.router, prefix="/api/pipeline", tags=["Pipeline"])
app.include_router(chat.router, prefix="/api", tags=["Chat"])
app.include_router(prediction.router, prefix="/api/prediction", tags=["Prediction"])
app.include_router(artifacts.router, prefix="/api/artifacts", tags=["Artifacts"])
app.include_router(datasets.router, prefix="/api/datasets", tags=["Datasets"])

@app.get("/")
async def root():
    return {
        "message": "Autonomous Agentic ML Engineering Platform API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_url": "/health",
        "endpoints": {
            "pipeline_stream": "/api/pipeline/stream",
            "pipeline_execute": "/api/pipeline/execute-code",
            "chat_stream": "/api/chat",
            "prediction": "/api/prediction",
            "artifacts": "/api/artifacts",
            "datasets": "/api/datasets"
        }
    }
