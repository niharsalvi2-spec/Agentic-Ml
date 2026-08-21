from pydantic import BaseModel
from typing import Dict, Any, List, Optional

class TestingResult(BaseModel):
    agent_name: str = "testing"
    success: bool = True
    metadata: Dict[str, Any] = {}
