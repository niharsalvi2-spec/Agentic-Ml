from fastapi import APIRouter
router = APIRouter()
@router.get("/list")
def list_data(): return {"datasets": []}
