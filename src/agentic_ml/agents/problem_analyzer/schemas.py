from pydantic import BaseModel
from typing import Dict, Any, List, Optional

class ProblemAnalyzerResult(BaseModel):
    agent_name: str = "problem_analyzer"
    success: bool = True
    metadata: Dict[str, Any] = {}
