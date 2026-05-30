"""
CATCAT annotation backend — HF Space (FastAPI).

Flow:
  POST /annotate  — saves image + label to HF Dataset, triggers background
                    retraining every TRAIN_EVERY annotations, then deletes
                    images post-training (labels kept permanently).
  GET  /stats     — returns counts, training status, model version.

Required Space secrets:
  HF_TOKEN      write-capable token
  DATASET_REPO  e.g. davemost/catcat-annotations
  MODEL_REPO    e.g. davemost/catcat-model
  TRAIN_EVERY   (optional, default 5)
"""
import os, json, threading, tempfile, shutil, time, csv
from pathlib import Path
from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

HF_TOKEN    = os.environ.get("HF_TOKEN", "")
DATASET_REPO = os.environ.get("DATASET_REPO", "")
MODEL_REPO  = os.environ.get("MODEL_REPO", "")
TRAIN_EVERY = int(os.environ.get("TRAIN_EVERY", "5"))

api = HfApi(token=HF_TOKEN)
_lock = threading.Lock()
_is_training = False
_annotation_count = 0
_model_version = "initial"
_training_run_count = 0
_training_started_at = None


# ── State helpers ──────────────────────────────────────────────────────────────

def _load_state():
    global _annotation_count, _model_version, _training_run_count
    try:
        p = hf_hub_download(DATASET_REPO, "state.json", repo_type="dataset", token=HF_TOKEN)
        s = json.loads(Path(p).read_text())
        _annotation_count   = s.get("annotation_count", 0)
        _model_version      = s.get("model_version", "initial")
        _training_run_count = s.get("training_run_count", 0)
    except Exception:
        pass

def _save_state():
    api.upload_file(
        path_or_fileobj=json.dumps({
            "annotation_count": _annotation_count,
            "model_version": _model_version,
            "training_run_count": _training_run_count,
        }).encode(),
        path_in_repo="state.json",
        repo_id=DATASET_REPO, repo_type="dataset",
        commit_message="state update",
    )

def _append_history(map50):
    history = []
    try:
        p = hf_hub_download(DATASET_REPO, "history.json", repo_type="dataset", token=HF_TOKEN)
        history = json.loads(Path(p).read_text())
    except Exception:
        pass
    history.append({
        "timestamp": int(time.time()),
        "map50": map50,
        "annotation_count": _annotation_count,
        "training_run": _training_run_count + 1,
    })
    api.upload_file(
        path_or_fileobj=json.dumps(history).encode(),
        path_in_repo="history.json",
        repo_id=DATASET_REPO, repo_type="dataset",
        commit_message="history update",
    )

def _images_in_queue() -> int:
    try:
        files = list(api.list_repo_files(DATASET_REPO, repo_type="dataset"))
        return sum(1 for f in files if f.startswith("images/"))
    except Exception:
        return 0

def _get_map50():
    try:
        p = hf_hub_download(MODEL_REPO, "metrics.json", repo_type="model", token=HF_TOKEN)
        return json.loads(Path(p).read_text()).get("map50")
    except Exception:
        return None

_load_state()


# ── Endpoints ─────────────────────────────────────────────────────────────────

def to_yolo(b: dict) -> str:
    cx = (b["x1"] + b["x2"]) / 2
    cy = (b["y1"] + b["y2"]) / 2
    return f"0 {cx:.6f} {cy:.6f} {b['x2']-b['x1']:.6f} {b['y2']-b['y1']:.6f}"


@app.post("/annotate")
async def annotate(
    image: UploadFile = File(...),
    boxes: str = Form(...),
    image_name: str = Form(...),
):
    global _annotation_count, _is_training
    if not DATASET_REPO:
        raise HTTPException(503, "DATASET_REPO not configured")

    stem = image_name.rsplit(".", 1)[0]
    ext  = image_name.rsplit(".", 1)[-1] if "." in image_name else "jpg"
    img_bytes = await image.read()

    api.upload_file(
        path_or_fileobj=img_bytes,
        path_in_repo=f"images/{stem}.{ext}",
        repo_id=DATASET_REPO, repo_type="dataset",
        commit_message=f"image: {image_name}",
    )
    api.upload_file(
        path_or_fileobj="\n".join(to_yolo(b) for b in json.loads(boxes)).encode(),
        path_in_repo=f"labels/{stem}.txt",
        repo_id=DATASET_REPO, repo_type="dataset",
        commit_message=f"label: {image_name}",
    )

    _annotation_count += 1
    _save_state()

    if _annotation_count % TRAIN_EVERY == 0 and not _is_training:
        threading.Thread(target=_retrain, daemon=True).start()

    return {
        "success": True,
        "total_annotations": _annotation_count,
        "training": _is_training,
        "model_version": _model_version,
        "map50": _get_map50(),
    }


@app.get("/stats")
async def stats():
    elapsed = int(time.time() - _training_started_at) if _is_training and _training_started_at else None
    return {
        "total_annotations": _annotation_count,
        "images_in_queue": _images_in_queue(),
        "training": _is_training,
        "training_elapsed_s": elapsed,
        "training_run_count": _training_run_count,
        "model_version": _model_version,
        "map50": _get_map50(),
        "train_every": TRAIN_EVERY,
    }

@app.get("/history")
async def history():
    try:
        p = hf_hub_download(DATASET_REPO, "history.json", repo_type="dataset", token=HF_TOKEN)
        return json.loads(Path(p).read_text())
    except Exception:
        return []


# ── Background retraining ──────────────────────────────────────────────────────

def _retrain():
    global _is_training, _model_version
    if not _lock.acquire(blocking=False):
        return
    _is_training = True
    _training_started_at = time.time()
    tmpdir = None
    try:
        from ultralytics import YOLO

        tmpdir = Path(tempfile.mkdtemp())
        img_dir = tmpdir / "images"
        lbl_dir = tmpdir / "labels"
        img_dir.mkdir(); lbl_dir.mkdir()

        # Download all images + labels
        snapshot_download(
            repo_id=DATASET_REPO, repo_type="dataset",
            local_dir=str(tmpdir), token=HF_TOKEN,
            ignore_patterns=["state.json"],
        )

        images = list(img_dir.iterdir())
        if not images:
            return

        # dataset.yaml
        (tmpdir / "dataset.yaml").write_text(
            f"path: {tmpdir}\ntrain: images\nval: images\nnc: 1\nnames: ['ship']\n"
        )

        # Load current weights (continue from best.pt, fall back to base model)
        try:
            weights = hf_hub_download(
                MODEL_REPO, "best.pt", repo_type="model", token=HF_TOKEN
            )
        except Exception:
            weights = "yolov8n.pt"

        # Train
        model = YOLO(weights)
        model.train(
            data=str(tmpdir / "dataset.yaml"),
            epochs=5, imgsz=640,
            project=str(tmpdir), name="train", exist_ok=True,
        )

        trained_pt = tmpdir / "train" / "weights" / "best.pt"
        if not trained_pt.exists():
            return

        # Export ONNX
        YOLO(str(trained_pt)).export(format="onnx", imgsz=640, simplify=True, opset=12)
        onnx_path = trained_pt.with_suffix(".onnx")

        # Parse mAP50
        map50 = None
        csv_path = tmpdir / "train" / "results.csv"
        if csv_path.exists():
            rows = list(csv.DictReader(csv_path.open()))
            if rows:
                try: map50 = float(rows[-1]["metrics/mAP50(B)"])
                except Exception: pass

        # Push model + weights + metrics
        for local, remote in [
            (str(onnx_path), "model.onnx"),
            (str(trained_pt), "best.pt"),
        ]:
            api.upload_file(
                path_or_fileobj=local, path_in_repo=remote,
                repo_id=MODEL_REPO, repo_type="model",
                commit_message=f"retrain: {remote}",
            )
        if map50 is not None:
            api.upload_file(
                path_or_fileobj=json.dumps({"map50": map50}).encode(),
                path_in_repo="metrics.json",
                repo_id=MODEL_REPO, repo_type="model",
                commit_message="retrain: metrics",
            )

        # Delete images from dataset (labels kept)
        for f in api.list_repo_files(DATASET_REPO, repo_type="dataset"):
            if f.startswith("images/"):
                try:
                    api.delete_file(
                        path_in_repo=f,
                        repo_id=DATASET_REPO, repo_type="dataset",
                        commit_message=f"cleanup: {f}",
                    )
                except Exception:
                    pass

        _model_version = str(int(time.time()))
        _training_run_count += 1
        _append_history(map50)
        _save_state()

    except Exception as e:
        print(f"[retrain error] {e}")
    finally:
        _is_training = False
        _lock.release()
        if tmpdir:
            shutil.rmtree(str(tmpdir), ignore_errors=True)
