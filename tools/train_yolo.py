"""
train_yolo.py — Train a YOLOv8 model for drone detection.

Usage:
    python tools/train_yolo.py --data dataset_yolo/dataset.yaml
    python tools/train_yolo.py --data dataset_yolo/dataset.yaml --model yolov8s.pt --epochs 200
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Train YOLOv8 drone detector")
    p.add_argument("--data", type=str, default="dataset_yolo/dataset.yaml",
                   help="Path to dataset.yaml")
    p.add_argument("--model", type=str, default="yolov8n.pt",
                   help="Pretrained model (yolov8n/s/m/l/x.pt)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=960,
                   help="Training image size (default: 960, matching RGB capture)")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", type=str, default="",
                   help="Device: '' for auto, '0' for GPU 0, 'cpu' for CPU")
    p.add_argument("--project", type=str, default="runs/train",
                   help="Project directory for results")
    p.add_argument("--name", type=str, default="drone_detector",
                   help="Experiment name")
    p.add_argument("--resume", action="store_true",
                   help="Resume training from last checkpoint")
    p.add_argument("--workers", type=int, default=4,
                   help="Dataloader workers (default: 4, use 0 if CUDA pin-memory errors)")
    args = p.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        return

    data_path = Path(args.data).resolve()
    if not data_path.exists():
        print(f"ERROR: {data_path} not found. Run generate_yolo_labels.py first.")
        return

    model = YOLO(args.model)

    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device or None,
        project=args.project,
        name=args.name,
        exist_ok=True,
        resume=args.resume,
        patience=20,
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
        workers=args.workers,
    )

    best = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\nTraining complete. Best model: {best.resolve()}")
    print(f"Results dir: {(Path(args.project) / args.name).resolve()}")


if __name__ == "__main__":
    main()
