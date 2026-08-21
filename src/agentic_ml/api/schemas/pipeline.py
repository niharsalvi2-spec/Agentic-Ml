from pydantic import BaseModel
class PipelineRequest(BaseModel):
    task: str
    dataset_path: str = ""
