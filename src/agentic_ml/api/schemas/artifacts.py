from pydantic import BaseModel
class ArtifactInfo(BaseModel):
    name: str
    path: str
