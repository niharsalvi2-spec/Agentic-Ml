from pydantic import BaseModel
from typing import Dict, Any, List, Optional

class DataCollectorResult(BaseModel):
    agent_name: str = "data_collector"
    success: bool = True
    metadata: Dict[str, Any] = {}
