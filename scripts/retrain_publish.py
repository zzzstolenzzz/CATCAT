"""Export, write metrics, and upload to HF Hub after retraining."""
import os, sys, csv, json
from pathlib import Path

weights = Path("runs/ship/weights/best.pt")
if not weights.exists():
    print("No weights found — training was skipped. Nothing to publish.")
    sys.exit(0)

# Export to ONNX
from ultralytics import YOLO
YOLO(str(weights)).export(format="onnx", imgsz=640, simplify=True, opset=12)
onnx_path = weights.with_suffix(".onnx")

# Parse metrics
map50 = None
csv_path = Path("runs/ship/results.csv")
if csv_path.exists():
    rows = list(csv.DictReader(csv_path.open()))
    if rows:
        try:
            map50 = float(rows[-1]["metrics/mAP50(B)"])
        except Exception:
            pass

metrics = {"map50": map50}
metrics_path = weights.parent / "metrics.json"
metrics_path.write_text(json.dumps(metrics))
print(f"mAP50: {map50}")

# Upload to HF Hub
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
model_repo   = os.environ["MODEL_REPO"]
dataset_repo = os.environ["DATASET_REPO"]

for local, remote in [(str(onnx_path), "model.onnx"), (str(weights), "best.pt")]:
    api.upload_file(path_or_fileobj=local, path_in_repo=remote,
                    repo_id=model_repo, repo_type="model",
                    commit_message=f"Retrain: {remote}")

api.upload_file(path_or_fileobj=str(metrics_path), path_in_repo="metrics.json",
                repo_id=dataset_repo, repo_type="dataset",
                commit_message="Retrain: metrics")

print("Published successfully.")
