import csv
import shutil
import sys
import threading
from pathlib import Path

from ultralytics import YOLO

APP_ROOT = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
RESOURCE_ROOT = Path(sys._MEIPASS) if getattr(sys, "frozen", False) else APP_ROOT
WEIGHTS_PATH = APP_ROOT / "data/weights/best.pt"
BASE_MODEL = str(RESOURCE_ROOT / "yolov8n.pt")
COCO_BOAT_CLASS = 8  # COCO dataset class index for "boat"


class ShipDetector:
    def __init__(self, on_status=None, on_train_complete=None, on_metrics=None):
        self.on_status = on_status or (lambda msg: None)
        self.on_train_complete = on_train_complete or (lambda: None)
        self.on_metrics = on_metrics or (lambda v: None)
        self.model = self._load_model()
        self._training = False
        self._retrain_pending = False

    def _load_model(self):
        if WEIGHTS_PATH.exists():
            return YOLO(str(WEIGHTS_PATH))
        return YOLO(BASE_MODEL)

    def detect(self, image_path):
        """Return the single largest detected ship as [(x1, y1, x2, y2)]."""
        results = self.model(str(image_path), verbose=False, iou=0.35)[0]
        boxes = []
        for box in results.boxes:
            cls = int(box.cls)
            # Custom model knows only "ship" (class 0); base COCO model uses class 8 = boat
            if WEIGHTS_PATH.exists() or cls == COCO_BOAT_CLASS:
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                boxes.append((x1, y1, x2, y2))
        if not boxes:
            return []
        # Keep only the largest box — the main (foreground) ship
        return [max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))]

    def train_async(self, data_yaml):
        if self._training:
            self._retrain_pending = True
            self.on_status("Training in progress — correction queued for next run.")
            return
        self._training = True
        threading.Thread(target=self._run_training, args=(data_yaml,), daemon=True).start()

    def _run_training(self, data_yaml):
        self.on_status("Training on CPU (may take a few minutes)…")
        try:
            base = str(WEIGHTS_PATH) if WEIGHTS_PATH.exists() else BASE_MODEL
            project_dir = Path(data_yaml).parent.resolve() / "runs"
            model = YOLO(base)
            model.train(
                data=data_yaml,
                epochs=5,
                imgsz=640,
                device="cpu",
                project=str(project_dir),
                name="ship",
                exist_ok=True,
                verbose=False,
                workers=0,  # Required on Windows to avoid multiprocessing errors
            )
            best = project_dir / "ship" / "weights" / "best.pt"
            if best.exists():
                WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(best, WEIGHTS_PATH)
                self.model = YOLO(str(WEIGHTS_PATH))
                self.on_status("Training complete — model updated with your corrections.")
                map50 = self._read_map50(project_dir)
                if map50 is not None:
                    self.on_metrics(map50)
            else:
                self.on_status("Training finished but no weights were produced (need more labeled images).")
        except Exception as exc:
            self.on_status(f"Training error: {exc}")
        finally:
            self._training = False
            self.on_train_complete()
            if self._retrain_pending:
                self._retrain_pending = False
                self.train_async(data_yaml)

    def _read_map50(self, project_dir):
        csv_path = project_dir / "ship" / "results.csv"
        if not csv_path.exists():
            return None
        try:
            with open(csv_path) as f:
                rows = list(csv.DictReader(f))
            if not rows:
                return None
            last = rows[-1]
            for k, v in last.items():
                if "mAP50" in k and "95" not in k:
                    return float(v.strip())
        except Exception:
            return None
