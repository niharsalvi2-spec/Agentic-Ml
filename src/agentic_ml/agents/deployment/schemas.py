from pydantic import BaseModel
from typing import Dict, Any, List, Optional

class DeploymentResult(BaseModel):
    agent_name: str = "deployment"
    success: bool = True
    metadata: Dict[str, Any] = {}
