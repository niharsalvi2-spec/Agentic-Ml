from pydantic import BaseModel
class MemoryRecord(BaseModel):
    namespace: str
    key: str
    value: dict
