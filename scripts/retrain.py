"""GitHub Actions retraining script."""
import os, sys, csv, json
from pathlib import Path

data_dir = Path(os.getcwd()) / "data"
img_dir  = data_dir / "images"
images   = list(img_dir.glob("*")) if img_dir.exists() else []

print(f"data_dir : {data_dir}")
print(f"images   : {len(images)}")

if not images:
    print("No images in dataset — nothing to train on. Exiting.")
    sys.exit(0)

yaml_path = data_dir / "dataset.yaml"
yaml_path.write_text(
    f"path: {data_dir}\ntrain: images\nval: images\nnc: 1\nnames:\n- ship\n"
)
print(f"dataset.yaml written:\n{yaml_path.read_text()}")

from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(data=str(yaml_path), epochs=10, imgsz=640, project="runs", name="ship")
