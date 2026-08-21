from pydantic import BaseModel
from typing import Dict, Any, List, Optional

class ModelBuildingResult(BaseModel):
    agent_name: str = "model_building"
    success: bool = True
    metadata: Dict[str, Any] = {}
