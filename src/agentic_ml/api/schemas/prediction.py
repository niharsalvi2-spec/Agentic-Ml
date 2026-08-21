from pydantic import BaseModel
from typing import List, Any
class PredictionRequest(BaseModel):
    data: List[Any]
