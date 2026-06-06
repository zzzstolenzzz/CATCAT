"""
CATCAT annotation backend — HF Space (FastAPI).

Three tiers:
  World  POST /annotate       — public, no auth
  Team   POST /team/annotate  — private, requires X-Team-Key header
  Local  (no server call)     — client-side only

Required Space secrets:
  HF_TOKEN           write-capable token
  DATASET_REPO       e.g. davemost/catcat-annotations        (public)
  MODEL_REPO         e.g. davemost/catcat-model              (public)
  TEAM_KEY           shared secret for team members
  TEAM_DATASET_REPO  e.g. davemost/catcat-team-annotations   (private)
  TEAM_MODEL_REPO    e.g. davemost/catcat-team-model         (private)
  TRAIN_EVERY        optional, default 5
"""
import os, json, threading, tempfile, shutil, time, csv
from pathlib import Path
from fastapi import FastAPI, Form, UploadFile, File, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import HfApi, hf_hub_download, snapshot_download, CommitOperationAdd, CommitOperationDelete
from typing import Optional

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
    allow_headers=["*", "X-Team-Key"],
)

# ── Config ────────────────────────────────────────────────────────────────────

HF_TOKEN          = os.environ.get("HF_TOKEN", "")
DATASET_REPO      = os.environ.get("DATASET_REPO", "")
MODEL_REPO        = os.environ.get("MODEL_REPO", "")
TEAM_KEY          = os.environ.get("TEAM_KEY", "")
TEAM_DATASET_REPO = os.environ.get("TEAM_DATASET_REPO", "")
TEAM_MODEL_REPO   = os.environ.get("TEAM_MODEL_REPO", "")
TRAIN_EVERY       = int(os.environ.get("TRAIN_EVERY", "5"))

api = HfApi(token=HF_TOKEN)

# ── World state ───────────────────────────────────────────────────────────────

_lock              = threading.Lock()
_is_training       = False
_annotation_count  = 0
_model_version     = "initial"
_training_run_count = 0
_training_started_at = None
_training_progress = {"epoch": 0, "epochs": 5, "loss": None, "map50": None}

# ── Team state ────────────────────────────────────────────────────────────────

_team_lock              = threading.Lock()
_team_is_training       = False
_team_annotation_count  = 0
_team_model_version     = "initial"
_team_training_run_count = 0
_team_training_started_at = None
_team_training_progress = {"epoch": 0, "epochs": 5, "loss": None, "map50": None}


# ── Helpers ───────────────────────────────────────────────────────────────────

def to_yolo(b: dict) -> str:
    cx = (b["x1"] + b["x2"]) / 2
    cy = (b["y1"] + b["y2"]) / 2
    return f"0 {cx:.6f} {cy:.6f} {b['x2']-b['x1']:.6f} {b['y2']-b['y1']:.6f}"

def _load_state(repo, prefix=""):
    """Load annotation_count, model_version, training_run_count from repo."""
    try:
        p = hf_hub_download(repo, "state.json", repo_type="dataset", token=HF_TOKEN)
        return json.loads(Path(p).read_text())
    except Exception:
        return {}

def _save_state_to(repo, count, version, runs):
    api.upload_file(
        path_or_fileobj=json.dumps({
            "annotation_count": count,
            "model_version": version,
            "training_run_count": runs,
        }).encode(),
        path_in_repo="state.json",
        repo_id=repo, repo_type="dataset",
        commit_message="state update",
    )

def _finalize_dataset(repo, images_to_delete, map50, image_stats, count, version, runs, timing=None):
    """Delete training images, update history and state — all in one commit."""
    history = []
    try:
        p = hf_hub_download(repo, "history.json", repo_type="dataset", token=HF_TOKEN)
        history = json.loads(Path(p).read_text())
    except Exception:
        pass
    entry = {"timestamp": int(time.time()), "map50": map50,
             "annotation_count": count, "training_run": runs}
    if timing:
        entry.update(timing)
    if image_stats:
        entry["images"] = image_stats
    history.append(entry)
    ops = [CommitOperationDelete(f) for f in images_to_delete]
    ops.append(CommitOperationAdd(path_in_repo="history.json",
                                  path_or_fileobj=json.dumps(history).encode()))
    ops.append(CommitOperationAdd(path_in_repo="state.json",
                                  path_or_fileobj=json.dumps({
                                      "annotation_count": count,
                                      "model_version": version,
                                      "training_run_count": runs,
                                  }).encode()))
    api.create_commit(repo_id=repo, repo_type="dataset", token=HF_TOKEN,
                      commit_message="retrain complete", operations=ops)

def _images_in_queue_for(repo) -> int:
    try:
        return sum(1 for f in api.list_repo_files(repo, repo_type="dataset")
                   if f.startswith("images/"))
    except Exception:
        return 0

def _get_map50_from(repo, repo_type="model") -> Optional[float]:
    try:
        p = hf_hub_download(repo, "metrics.json", repo_type=repo_type, token=HF_TOKEN)
        return json.loads(Path(p).read_text()).get("map50")
    except Exception:
        return None

def _upload_annotation(image_bytes, boxes_json, stem, ext, dataset_repo, state: dict, corrected: bool = False):
    """Upload image, label, state, and correction flag in a single commit."""
    corrections = {}
    try:
        p = hf_hub_download(dataset_repo, "corrections.json", repo_type="dataset", token=HF_TOKEN)
        corrections = json.loads(Path(p).read_text())
    except Exception:
        pass
    corrections[stem] = corrected
    api.create_commit(
        repo_id=dataset_repo, repo_type="dataset",
        commit_message=f"annotate: {stem}",
        operations=[
            CommitOperationAdd(path_in_repo=f"images/{stem}.{ext}", path_or_fileobj=image_bytes),
            CommitOperationAdd(path_in_repo=f"labels/{stem}.txt",
                               path_or_fileobj="\n".join(to_yolo(b) for b in boxes_json).encode()),
            CommitOperationAdd(path_in_repo="state.json",
                               path_or_fileobj=json.dumps(state).encode()),
            CommitOperationAdd(path_in_repo="corrections.json",
                               path_or_fileobj=json.dumps(corrections).encode()),
        ],
        token=HF_TOKEN,
    )

def _check_team_key(key: Optional[str]):
    if not TEAM_KEY:
        raise HTTPException(503, "Team tier not configured")
    if key != TEAM_KEY:
        raise HTTPException(403, "Invalid team key")


# ── Boot: load saved state ────────────────────────────────────────────────────

def _boot():
    global _annotation_count, _model_version, _training_run_count
    global _team_annotation_count, _team_model_version, _team_training_run_count

    if DATASET_REPO:
        s = _load_state(DATASET_REPO)
        _annotation_count   = s.get("annotation_count", 0)
        _model_version      = s.get("model_version", "initial")
        _training_run_count = s.get("training_run_count", 0)

    if TEAM_DATASET_REPO:
        s = _load_state(TEAM_DATASET_REPO)
        _team_annotation_count   = s.get("annotation_count", 0)
        _team_model_version      = s.get("model_version", "initial")
        _team_training_run_count = s.get("training_run_count", 0)

_boot()


# ── World endpoints ───────────────────────────────────────────────────────────

@app.post("/annotate")
async def annotate(
    image: UploadFile = File(...),
    boxes: str = Form(...),
    image_name: str = Form(...),
    corrected: str = Form("0"),
):
    global _annotation_count, _is_training
    if not DATASET_REPO:
        raise HTTPException(503, "DATASET_REPO not configured")

    stem = image_name.rsplit(".", 1)[0]
    ext  = image_name.rsplit(".", 1)[-1] if "." in image_name else "jpg"
    _annotation_count += 1
    _upload_annotation(await image.read(), json.loads(boxes), stem, ext, DATASET_REPO,
                       {"annotation_count": _annotation_count, "model_version": _model_version,
                        "training_run_count": _training_run_count}, corrected == "1")

    if _annotation_count % TRAIN_EVERY == 0 and not _is_training:
        threading.Thread(target=_retrain_world, daemon=True).start()

    return {"success": True, "total_annotations": _annotation_count,
            "training": _is_training, "model_version": _model_version,
            "map50": _get_map50_from(MODEL_REPO)}


@app.get("/stats")
async def stats():
    elapsed = int(time.time() - _training_started_at) if _is_training and _training_started_at else None
    return {
        "total_annotations": _annotation_count,
        "images_in_queue": _images_in_queue_for(DATASET_REPO),
        "training": _is_training,
        "training_elapsed_s": elapsed,
        "training_progress": _training_progress,
        "training_run_count": _training_run_count,
        "model_version": _model_version,
        "map50": _get_map50_from(MODEL_REPO),
        "train_every": TRAIN_EVERY,
    }

@app.get("/history")
async def history():
    try:
        p = hf_hub_download(DATASET_REPO, "history.json", repo_type="dataset", token=HF_TOKEN)
        return json.loads(Path(p).read_text())
    except Exception:
        return []


# ── Team endpoints ────────────────────────────────────────────────────────────

@app.post("/team/annotate")
async def team_annotate(
    image: UploadFile = File(...),
    boxes: str = Form(...),
    image_name: str = Form(...),
    corrected: str = Form("0"),
    x_team_key: Optional[str] = Header(None),
):
    global _team_annotation_count, _team_is_training
    _check_team_key(x_team_key)
    if not TEAM_DATASET_REPO:
        raise HTTPException(503, "TEAM_DATASET_REPO not configured")

    stem = image_name.rsplit(".", 1)[0]
    ext  = image_name.rsplit(".", 1)[-1] if "." in image_name else "jpg"
    _team_annotation_count += 1
    _upload_annotation(await image.read(), json.loads(boxes), stem, ext, TEAM_DATASET_REPO,
                       {"annotation_count": _team_annotation_count,
                        "model_version": _team_model_version,
                        "training_run_count": _team_training_run_count}, corrected == "1")

    if _team_annotation_count % TRAIN_EVERY == 0 and not _team_is_training:
        threading.Thread(target=_retrain_team, daemon=True).start()

    return {"success": True, "total_annotations": _team_annotation_count,
            "training": _team_is_training, "model_version": _team_model_version,
            "map50": _get_map50_from(TEAM_MODEL_REPO)}


@app.get("/team/stats")
async def team_stats(x_team_key: Optional[str] = Header(None)):
    _check_team_key(x_team_key)
    elapsed = int(time.time() - _team_training_started_at) \
              if _team_is_training and _team_training_started_at else None
    return {
        "total_annotations": _team_annotation_count,
        "images_in_queue": _images_in_queue_for(TEAM_DATASET_REPO),
        "training": _team_is_training,
        "training_elapsed_s": elapsed,
        "training_progress": _team_training_progress,
        "training_run_count": _team_training_run_count,
        "model_version": _team_model_version,
        "map50": _get_map50_from(TEAM_MODEL_REPO),
        "train_every": TRAIN_EVERY,
    }

@app.get("/team/history")
async def team_history(x_team_key: Optional[str] = Header(None)):
    _check_team_key(x_team_key)
    try:
        p = hf_hub_download(TEAM_DATASET_REPO, "history.json",
                            repo_type="dataset", token=HF_TOKEN)
        return json.loads(Path(p).read_text())
    except Exception:
        return []


# ── World retraining ──────────────────────────────────────────────────────────

def _retrain_world():
    global _is_training, _model_version, _training_run_count, _training_started_at
    if not _lock.acquire(blocking=False):
        return
    _is_training = True
    _training_started_at = time.time()
    tmpdir = None
    try:
        map50, image_stats, images_to_delete, timing = _run_training(
            dataset_repo=DATASET_REPO,
            model_repo=MODEL_REPO,
            base_weights_repo=MODEL_REPO,
            progress_dict=_training_progress,
        )
        _model_version = str(int(time.time()))
        _training_run_count += 1
        _finalize_dataset(DATASET_REPO, images_to_delete, map50, image_stats,
                          _annotation_count, _model_version, _training_run_count, timing)
    except Exception as e:
        print(f"[retrain world error] {e}")
    finally:
        _is_training = False
        _lock.release()


# ── Team retraining ───────────────────────────────────────────────────────────

def _retrain_team():
    """Fine-tune on team data, starting from latest WORLD best.pt."""
    global _team_is_training, _team_model_version, _team_training_run_count
    global _team_training_started_at
    if not _team_lock.acquire(blocking=False):
        return
    _team_is_training = True
    _team_training_started_at = time.time()
    try:
        map50, image_stats, images_to_delete, timing = _run_training(
            dataset_repo=TEAM_DATASET_REPO,
            model_repo=TEAM_MODEL_REPO,
            base_weights_repo=MODEL_REPO,
            progress_dict=_team_training_progress,
        )
        _team_model_version = str(int(time.time()))
        _team_training_run_count += 1
        _finalize_dataset(TEAM_DATASET_REPO, images_to_delete, map50, image_stats,
                          _team_annotation_count, _team_model_version, _team_training_run_count, timing)
    except Exception as e:
        print(f"[retrain team error] {e}")
    finally:
        _team_is_training = False
        _team_lock.release()


# ── Shared training logic ─────────────────────────────────────────────────────

def _run_training(dataset_repo, model_repo, base_weights_repo, progress_dict):
    from ultralytics import YOLO

    tmpdir = Path(tempfile.mkdtemp())
    try:
        img_dir = tmpdir / "images"
        lbl_dir = tmpdir / "labels"
        img_dir.mkdir(); lbl_dir.mkdir()

        snapshot_download(
            repo_id=dataset_repo, repo_type="dataset",
            local_dir=str(tmpdir), token=HF_TOKEN,
            ignore_patterns=["state.json", "history.json"],
        )

        if not any(img_dir.iterdir()):
            print(f"[training] No images in {dataset_repo}, skipping")
            return None, [], [], {}

        (tmpdir / "dataset.yaml").write_text(
            f"path: {tmpdir}\ntrain: images\nval: images\nnc: 1\nnames: ['ship']\n"
        )

        try:
            weights = hf_hub_download(base_weights_repo, "best.pt",
                                       repo_type="model", token=HF_TOKEN)
        except Exception:
            weights = "yolov8n.pt"

        EPOCHS = 5
        progress_dict.update({"epoch": 0, "epochs": EPOCHS, "loss": None, "map50": None})
        epoch_times = []
        _epoch_start = [None]

        def on_epoch_start(trainer):
            _epoch_start[0] = time.time()

        def on_epoch_end(trainer):
            if _epoch_start[0]:
                epoch_times.append(round(time.time() - _epoch_start[0], 1))
            progress_dict.update({
                "epoch": trainer.epoch + 1,
                "epochs": trainer.epochs,
                "loss": round(float(trainer.loss), 4) if trainer.loss is not None else None,
                "map50": round(float(trainer.metrics.get("metrics/mAP50(B)", 0)), 4),
            })

        model = YOLO(weights)
        model.add_callback("on_train_epoch_start", on_epoch_start)
        model.add_callback("on_train_epoch_end", on_epoch_end)
        train_start = time.time()
        model.train(data=str(tmpdir / "dataset.yaml"), epochs=EPOCHS,
                    imgsz=640, project=str(tmpdir), name="train", exist_ok=True)
        total_seconds = round(time.time() - train_start)

        trained_pt = tmpdir / "train" / "weights" / "best.pt"
        if not trained_pt.exists():
            return None, [], [], {}

        YOLO(str(trained_pt)).export(format="onnx", imgsz=640, simplify=True, opset=12)
        onnx_path = trained_pt.with_suffix(".onnx")

        map50 = None
        csv_path = tmpdir / "train" / "results.csv"
        if csv_path.exists():
            rows = list(csv.DictReader(csv_path.open()))
            if rows:
                try: map50 = float(rows[-1]["metrics/mAP50(B)"])
                except Exception: pass

        # Load correction flags (already in tmpdir from snapshot_download)
        corrections = {}
        try:
            corrections = json.loads((tmpdir / "corrections.json").read_text())
        except Exception:
            pass

        # Per-image predictions
        image_stats = []
        try:
            preds = YOLO(str(trained_pt)).predict(
                source=str(img_dir), imgsz=640, conf=0.1, verbose=False, save=False
            )
            for r in preds:
                confs = r.boxes.conf.tolist() if r.boxes is not None and len(r.boxes) > 0 else []
                stem = Path(r.path).stem
                image_stats.append({
                    "name": Path(r.path).name,
                    "detections": len(confs),
                    "max_conf": round(max(confs), 3) if confs else 0.0,
                    "avg_conf": round(sum(confs) / len(confs), 3) if confs else 0.0,
                    "corrected": corrections.get(stem, None),
                })
        except Exception as e:
            print(f"[per-image stats error] {e}")

        # Batch model files into one commit
        model_ops = [
            CommitOperationAdd(path_in_repo="model.onnx", path_or_fileobj=onnx_path.read_bytes()),
            CommitOperationAdd(path_in_repo="best.pt",    path_or_fileobj=trained_pt.read_bytes()),
        ]
        if map50 is not None:
            model_ops.append(CommitOperationAdd(path_in_repo="metrics.json",
                                                path_or_fileobj=json.dumps({"map50": map50}).encode()))
        api.create_commit(repo_id=model_repo, repo_type="model", token=HF_TOKEN,
                          commit_message="retrain: model", operations=model_ops)

        # Collect images to delete (caller batches with history+state in one commit)
        images_to_delete = [f for f in api.list_repo_files(dataset_repo, repo_type="dataset")
                            if f.startswith("images/")]

        avg_epoch_s = round(sum(epoch_times) / len(epoch_times), 1) if epoch_times else None
        timing = {"total_seconds": total_seconds, "avg_epoch_s": avg_epoch_s}
        return map50, image_stats, images_to_delete, timing

    finally:
        shutil.rmtree(str(tmpdir), ignore_errors=True)
