"""
Export the fine-tuned YOLOv8 model to ONNX for browser inference.

Usage:
  python scripts/export_onnx.py
  python scripts/export_onnx.py --weights runs/ship/weights/best.pt

Then upload the resulting .onnx file to your HF Hub model repo:
  huggingface-cli upload YOUR_HF_USERNAME/catcat-model runs/ship/weights/best.onnx model.onnx
"""
import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="runs/ship/weights/best.pt",
                        help="Path to trained .pt weights")
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    print(f"Exporting {weights} to ONNX…")
    model = YOLO(str(weights))
    model.export(format="onnx", imgsz=640, simplify=True, opset=12)

    onnx_path = weights.with_suffix(".onnx")
    print(f"\nExported: {onnx_path}")
    print("\nNext steps:")
    print(f"  huggingface-cli upload YOUR_HF_USERNAME/catcat-model {onnx_path} model.onnx")


if __name__ == "__main__":
    main()
