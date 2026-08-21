from pydantic import BaseModel
from typing import Dict, Any, List, Optional

class PreprocessingResult(BaseModel):
    agent_name: str = "preprocessing"
    success: bool = True
    metadata: Dict[str, Any] = {}
