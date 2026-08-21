from pydantic import BaseModel
from typing import Dict, Any, List, Optional

class FeatureSelectionResult(BaseModel):
    agent_name: str = "feature_selection"
    success: bool = True
    metadata: Dict[str, Any] = {}
