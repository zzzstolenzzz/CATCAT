import shutil
import sys
import threading
import tkinter as tk
import yaml
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from detector import ShipDetector

APP_ROOT = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
IMAGES_DIR = APP_ROOT / "data/images"
LABELS_DIR = APP_ROOT / "data/labels"
DATASET_YAML = APP_ROOT / "data/dataset.yaml"
IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.webp")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CATCAT — Computer-Assisted Target Classification and Annotation Tool")
        self.geometry("1000x800")
        self.minsize(700, 550)

        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        LABELS_DIR.mkdir(parents=True, exist_ok=True)
        self._write_dataset_yaml()

        self.detector = ShipDetector(on_status=self._set_status, on_metrics=self._on_metrics)

        self.image_paths = []
        self.idx = 0
        self.pil_image = None
        self.photo = None
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.boxes = []          # (x1, y1, x2, y2) in original image pixels
        self.has_corrections = False
        self._drag_start = None
        self._drag_rect = None
        self.session_count = 0
        self.last_map50 = None

        self._build_ui()

    # ── Setup ──────────────────────────────────────────────────────────────────

    def _write_dataset_yaml(self):
        with open(DATASET_YAML, "w") as f:
            yaml.dump(
                {
                    "path": str(Path("data").resolve()),
                    "train": "images",
                    "val": "images",
                    "nc": 1,
                    "names": ["ship"],
                },
                f,
                default_flow_style=False,
            )

    def _build_ui(self):
        toolbar = ttk.Frame(self)
        toolbar.pack(fill=tk.X, padx=6, pady=4)

        ttk.Button(toolbar, text="Load Images", command=self._load_images).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Load Folder", command=self._load_folder).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Import Existing Data", command=self._import_data).pack(side=tk.LEFT, padx=2)

        self._counter_var = tk.StringVar()
        ttk.Label(toolbar, textvariable=self._counter_var).pack(side=tk.RIGHT, padx=10)

        self._canvas = tk.Canvas(self, bg="#111", cursor="crosshair")
        self._canvas.pack(fill=tk.BOTH, expand=True, padx=6)
        self._canvas.bind("<ButtonPress-1>", self._on_press)
        self._canvas.bind("<B1-Motion>", self._on_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_release)
        self._canvas.bind("<Motion>", self._on_motion)
        self._canvas.bind("<Leave>", lambda _e: self._canvas.delete("crosshair"))
        self.bind("<Configure>", lambda _e: self._redraw())
        self.bind("<Return>", lambda _e: self._accept())

        acts = ttk.Frame(self)
        acts.pack(fill=tk.X, padx=6, pady=4)

        self._accept_btn = ttk.Button(acts, text="Accept  [Enter]", command=self._accept, state=tk.DISABLED)
        self._accept_btn.pack(side=tk.LEFT, padx=2)

        self._clear_btn = ttk.Button(acts, text="Clear Boxes", command=self._clear_boxes, state=tk.DISABLED)
        self._clear_btn.pack(side=tk.LEFT, padx=2)

        self._back_btn = ttk.Button(acts, text="Back", command=self._prev, state=tk.DISABLED)
        self._back_btn.pack(side=tk.LEFT, padx=2)

        ttk.Button(acts, text="Skip", command=self._next).pack(side=tk.LEFT, padx=2)

        metrics_frame = ttk.Frame(self, relief=tk.SUNKEN)
        metrics_frame.pack(fill=tk.X, padx=6, pady=(0, 2))

        self._count_label = tk.Label(metrics_frame, text="Session: 0 images processed", anchor=tk.W, padx=6)
        self._count_label.pack(side=tk.LEFT)

        ttk.Separator(metrics_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=2)

        self._accuracy_label = tk.Label(metrics_frame, text="Model Accuracy (mAP50): —", anchor=tk.W)
        self._accuracy_label.pack(side=tk.LEFT)
        self._accuracy_label_default_fg = self._accuracy_label.cget("foreground")

        self._status_var = tk.StringVar(value="Load images to begin.")
        ttk.Label(self, textvariable=self._status_var, relief=tk.SUNKEN, anchor=tk.W).pack(
            fill=tk.X, padx=6, pady=(0, 4)
        )

    # ── File loading ───────────────────────────────────────────────────────────

    def _already_reviewed(self, path):
        return (LABELS_DIR / (path.stem + ".txt")).exists()

    def _load_images(self):
        paths = filedialog.askopenfilenames(
            filetypes=[("Images", " ".join(IMAGE_EXTS))]
        )
        if not paths:
            return
        all_paths = [Path(p) for p in paths]
        self.image_paths = [p for p in all_paths if not self._already_reviewed(p)]
        skipped = len(all_paths) - len(self.image_paths)
        if skipped:
            self._set_status(f"Skipped {skipped} already-reviewed image(s).")
        if self.image_paths:
            self.idx = 0
            self._show_current()
        else:
            messagebox.showinfo("All reviewed", "All selected images have already been reviewed.")

    def _load_folder(self):
        folder = filedialog.askdirectory()
        if not folder:
            return
        all_paths = sorted(p for ext in IMAGE_EXTS for p in Path(folder).rglob(ext))
        self.image_paths = [p for p in all_paths if not self._already_reviewed(p)]
        skipped = len(all_paths) - len(self.image_paths)
        if not all_paths:
            messagebox.showinfo("No images", "No supported image files found in that folder.")
        elif not self.image_paths:
            messagebox.showinfo("All reviewed", f"All {len(all_paths)} images in this folder have already been reviewed.")
        else:
            if skipped:
                self._set_status(f"Loaded {len(self.image_paths)} unreviewed image(s), skipped {skipped} already reviewed.")
            self.idx = 0
            self._show_current()

    def _import_data(self):
        """Copy a YOLO-formatted dataset (images/ + labels/) into the training data folder."""
        folder = filedialog.askdirectory(title="Select dataset root (must contain images/ and labels/)")
        if not folder:
            return
        src = Path(folder)
        if not (src / "images").exists() or not (src / "labels").exists():
            messagebox.showerror(
                "Invalid dataset",
                "Selected folder must contain 'images' and 'labels' subdirectories.",
            )
            return
        count = 0
        for img in (src / "images").iterdir():
            if img.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tiff"):
                shutil.copy(img, IMAGES_DIR / img.name)
                lbl = src / "labels" / (img.stem + ".txt")
                if lbl.exists():
                    shutil.copy(lbl, LABELS_DIR / lbl.name)
                count += 1
        self._set_status(f"Imported {count} image(s) into training dataset.")

    # ── Display ────────────────────────────────────────────────────────────────

    def _show_current(self):
        if not self.image_paths:
            return
        path = self.image_paths[self.idx]
        self._counter_var.set(f"Image {self.idx + 1} of {len(self.image_paths)}")
        self._set_status(f"Detecting ships in {path.name}…")
        self.update_idletasks()

        self.pil_image = Image.open(path).convert("RGB")
        self.boxes = []
        self.has_corrections = False
        self._redraw()

        for btn in (self._accept_btn, self._clear_btn):
            btn.config(state=tk.NORMAL)
        self._back_btn.config(state=tk.NORMAL if self.idx > 0 else tk.DISABLED)

        saved = self._load_saved_boxes(path)
        if saved is not None:
            self.boxes = saved
            self._redraw()
            self._set_status(f"Showing saved annotation. Press Enter to move on or draw to change.")
        else:
            threading.Thread(target=self._run_detect, args=(path,), daemon=True).start()

    def _load_saved_boxes(self, path):
        label_file = LABELS_DIR / (path.stem + ".txt")
        if not label_file.exists():
            return None
        iw, ih = self.pil_image.size
        boxes = []
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    _, cx, cy, w, h = map(float, parts)
                    x1 = (cx - w / 2) * iw
                    y1 = (cy - h / 2) * ih
                    x2 = (cx + w / 2) * iw
                    y2 = (cy + h / 2) * ih
                    boxes.append((x1, y1, x2, y2))
        return boxes

    def _run_detect(self, path):
        boxes = self.detector.detect(path)
        self.after(0, lambda: self._on_detect_done(boxes))

    def _pad_box(self, box, pad=20):
        x1, y1, x2, y2 = box
        iw, ih = self.pil_image.size
        return (
            max(0.0, x1 - pad),
            max(0.0, y1 - pad),
            min(float(iw), x2 + pad),
            min(float(ih), y2 + pad),
        )

    def _on_detect_done(self, boxes):
        self.boxes = [self._pad_box(b) for b in boxes]
        self.has_corrections = False
        self._redraw()
        n = len(self.boxes)
        self._set_status(f"Found {n} ship{'s' if n != 1 else ''}. Press Enter to accept or draw to correct.")

    def _redraw(self):
        if not self.pil_image:
            return
        cw = self._canvas.winfo_width() or 900
        ch = self._canvas.winfo_height() or 650
        iw, ih = self.pil_image.size
        self.scale = min(cw / iw, ch / ih)
        nw, nh = int(iw * self.scale), int(ih * self.scale)
        self.offset_x = (cw - nw) // 2
        self.offset_y = (ch - nh) // 2

        resized = self.pil_image.resize((nw, nh), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(resized)
        self._canvas.delete("all")
        self._canvas.create_image(self.offset_x, self.offset_y, anchor=tk.NW, image=self.photo)

        # Green = YOLO prediction, Yellow = user correction
        color = "yellow" if self.has_corrections else "lime"
        for x1, y1, x2, y2 in self.boxes:
            self._canvas.create_rectangle(
                self.offset_x + x1 * self.scale,
                self.offset_y + y1 * self.scale,
                self.offset_x + x2 * self.scale,
                self.offset_y + y2 * self.scale,
                outline=color,
                width=2,
            )

    # ── Drawing ────────────────────────────────────────────────────────────────

    def _draw_crosshair(self, x, y):
        self._canvas.delete("crosshair")
        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        self._canvas.create_line(0, y, cw, y, fill="#355E3B", tags="crosshair")
        self._canvas.create_line(x, 0, x, ch, fill="#355E3B", tags="crosshair")

    def _on_motion(self, event):
        if self.pil_image:
            x, y = self._clamp_to_image(event.x, event.y)
            self._draw_crosshair(x, y)

    def _clamp_to_image(self, x, y):
        iw, ih = self.pil_image.size
        x = max(self.offset_x, min(x, self.offset_x + iw * self.scale))
        y = max(self.offset_y, min(y, self.offset_y + ih * self.scale))
        return x, y

    def _on_press(self, event):
        if self.pil_image:
            self.boxes = []
            self.has_corrections = True
            self._redraw()
            self._drag_start = self._clamp_to_image(event.x, event.y)

    def _on_drag(self, event):
        if not self._drag_start:
            return
        if self._drag_rect:
            self._canvas.delete(self._drag_rect)
        ex, ey = self._clamp_to_image(event.x, event.y)
        self._draw_crosshair(ex, ey)
        x0, y0 = self._drag_start
        self._drag_rect = self._canvas.create_rectangle(
            x0, y0, ex, ey, outline="red", width=2
        )

    def _on_release(self, event):
        if not self._drag_start:
            return
        if self._drag_rect:
            self._canvas.delete(self._drag_rect)
            self._drag_rect = None

        x0, y0 = self._drag_start
        self._drag_start = None
        ex, ey = self._clamp_to_image(event.x, event.y)
        cx1, cy1 = min(x0, ex), min(y0, ey)
        cx2, cy2 = max(x0, ex), max(y0, ey)

        if cx2 - cx1 < 8 or cy2 - cy1 < 8:
            return  # Ignore accidental micro-drags

        iw, ih = self.pil_image.size
        x1 = max(0.0, (cx1 - self.offset_x) / self.scale)
        y1 = max(0.0, (cy1 - self.offset_y) / self.scale)
        x2 = min(float(iw), (cx2 - self.offset_x) / self.scale)
        y2 = min(float(ih), (cy2 - self.offset_y) / self.scale)

        self.boxes.append((x1, y1, x2, y2))
        self.has_corrections = True
        self._redraw()

    def _clear_boxes(self):
        self.boxes = []
        self.has_corrections = True
        self._redraw()

    # ── Save / navigate ────────────────────────────────────────────────────────

    def _accept(self):
        if not self.pil_image:
            return
        if not self.boxes:
            if not messagebox.askyesno(
                "No ships marked",
                "No boxes drawn. Save this image as a negative example (no ships present)?",
            ):
                return
        self._save_annotation()
        self.session_count += 1
        self._count_label.config(text=f"Session: {self.session_count} image{'s' if self.session_count != 1 else ''} processed")
        if self.has_corrections:
            self.detector.train_async(str(DATASET_YAML))
        self._next()

    def _on_metrics(self, map50):
        pct = f"{map50 * 100:.1f}%"
        if self.last_map50 is None:
            color = self._accuracy_label_default_fg
        elif map50 >= self.last_map50:
            color = "green"
        else:
            color = "red"
        self.last_map50 = map50
        self.after(0, lambda: self._flash_accuracy(pct, color))

    def _flash_accuracy(self, pct, color):
        self._accuracy_label.config(text=f"Model Accuracy (mAP50): {pct}", foreground=color)
        self.after(3000, lambda: self._accuracy_label.config(foreground=self._accuracy_label_default_fg))

    def _save_annotation(self):
        src = self.image_paths[self.idx]
        dest = IMAGES_DIR / src.name
        if not dest.exists():
            shutil.copy(src, dest)
        iw, ih = self.pil_image.size
        with open(LABELS_DIR / (src.stem + ".txt"), "w") as f:
            for x1, y1, x2, y2 in self.boxes:
                cx = (x1 + x2) / 2 / iw
                cy = (y1 + y2) / 2 / ih
                w = (x2 - x1) / iw
                h = (y2 - y1) / ih
                f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    def _prev(self):
        if self.idx > 0:
            self.idx -= 1
            self._show_current()

    def _next(self):
        if self.idx < len(self.image_paths) - 1:
            self.idx += 1
            self._show_current()
        else:
            self._set_status("All images reviewed.")
            messagebox.showinfo("Done", "All images reviewed!")

    def _set_status(self, msg):
        self.after(0, lambda: self._status_var.set(msg))


if __name__ == "__main__":
    App().mainloop()
