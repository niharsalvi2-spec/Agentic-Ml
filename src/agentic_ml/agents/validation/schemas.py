from pydantic import BaseModel
from typing import Dict, Any, List, Optional

class ValidationResult(BaseModel):
    agent_name: str = "validation"
    success: bool = True
    metadata: Dict[str, Any] = {}
