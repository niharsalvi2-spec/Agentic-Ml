from pydantic import BaseModel
from typing import Dict, Any, List, Optional

class FeatureEngineeringResult(BaseModel):
    agent_name: str = "feature_engineering"
    success: bool = True
    metadata: Dict[str, Any] = {}
