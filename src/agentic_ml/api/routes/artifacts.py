from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from src.agentic_ml.core.constants import MODELS_DIR

router = APIRouter()

@router.get("/download/{filename}")
def download_artifact(filename: str):
    """
    Downloads the actual trained model.pkl or metadata file from artifacts/models/.
    """
    safe_filename = Path(filename).name
    filepath = MODELS_DIR / safe_filename
    
    if not filepath.exists():
        raise HTTPException(status_code=404, detail=f"Artifact '{safe_filename}' not found. Please run the pipeline first.")
        
    return FileResponse(
        path=str(filepath),
        filename=safe_filename,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{safe_filename}"',
            "Access-Control-Expose-Headers": "Content-Disposition"
        }
    )

@router.get("/metadata")
def get_model_metadata():
    """Returns the latest model metadata JSON."""
    meta_path = MODELS_DIR / "model_metadata.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="No metadata found.")
    import json
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)
