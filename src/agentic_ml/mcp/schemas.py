from pydantic import BaseModel
class MCPTaskRequest(BaseModel):
    task: str
