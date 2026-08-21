from pydantic import BaseModel
from typing import Dict, Any, List, Optional

class EdaResult(BaseModel):
    agent_name: str = "eda"
    success: bool = True
    metadata: Dict[str, Any] = {}
