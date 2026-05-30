"""
CATCAT annotation backend — runs as a Hugging Face Space (FastAPI).
Receives bounding box annotations from the browser UI and writes
YOLO-format label files to a HF Dataset repository.

Required environment variables (set in the Space's Secrets panel):
  HF_TOKEN       — a HF token with write access to DATASET_REPO
  DATASET_REPO   — e.g. "yourname/catcat-annotations"
"""
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from huggingface_hub import HfApi

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)

HF_TOKEN = os.environ.get("HF_TOKEN")
DATASET_REPO = os.environ.get("DATASET_REPO", "")
api = HfApi(token=HF_TOKEN)


class Box(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class AnnotationRequest(BaseModel):
    image_name: str
    boxes: List[Box]


def to_yolo(box: Box) -> str:
    cx = (box.x1 + box.x2) / 2
    cy = (box.y1 + box.y2) / 2
    w = box.x2 - box.x1
    h = box.y2 - box.y1
    return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


@app.post("/annotate")
async def annotate(req: AnnotationRequest):
    if not DATASET_REPO:
        raise HTTPException(status_code=503, detail="DATASET_REPO not configured")

    label_lines = "\n".join(to_yolo(b) for b in req.boxes)
    stem = req.image_name.rsplit(".", 1)[0]
    path_in_repo = f"labels/{stem}.txt"

    try:
        api.upload_file(
            path_or_fileobj=label_lines.encode(),
            path_in_repo=path_in_repo,
            repo_id=DATASET_REPO,
            repo_type="dataset",
            commit_message=f"annotation: {req.image_name}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    total = _count_labels()
    return {"success": True, "total_annotations": total, "map50": _latest_map50()}


@app.get("/stats")
async def stats():
    return {"total_annotations": _count_labels(), "map50": _latest_map50()}


def _count_labels() -> int:
    try:
        files = list(api.list_repo_files(DATASET_REPO, repo_type="dataset"))
        return sum(1 for f in files if f.startswith("labels/") and f.endswith(".txt"))
    except Exception:
        return 0


def _latest_map50() -> float | None:
    """Read mAP50 from a metrics file the retraining workflow writes back to the dataset."""
    try:
        from huggingface_hub import hf_hub_download
        import json
        path = hf_hub_download(DATASET_REPO, "metrics.json", repo_type="dataset", token=HF_TOKEN)
        with open(path) as f:
            return json.load(f).get("map50")
    except Exception:
        return None
