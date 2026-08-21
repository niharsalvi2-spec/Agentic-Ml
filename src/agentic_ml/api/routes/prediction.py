from fastapi import APIRouter
from src.agentic_ml.api.schemas.prediction import PredictionRequest
router = APIRouter()
@router.post("/predict")
def predict(req: PredictionRequest): return {"predictions": []}
