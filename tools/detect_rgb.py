"""
detect_rgb.py — Run YOLOv8 inference on the dataset and evaluate detection
performance vs ground-truth distance.

Usage:
    python tools/detect_rgb.py --model runs/train/drone_detector/weights/best.pt --dataset dataset
    python tools/detect_rgb.py --model runs/train/drone_detector/weights/best.pt --dataset dataset --save-images
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def load_telemetry(tele_path: Path) -> dict[int, dict]:
    """Load telemetry CSV and return {frame_idx: row_dict}."""
    rows: dict[int, dict] = {}
    with open(tele_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[int(row["frame"])] = row
    return rows


def iou(box1, box2):
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def yolo_to_xyxy(cx, cy, w, h, img_w, img_h):
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return [x1, y1, x2, y2]


def load_gt_label(label_path: Path, img_w: int, img_h: int):
    """Load YOLO ground-truth label and return xyxy box or None."""
    if not label_path.exists():
        return None
    text = label_path.read_text().strip()
    if not text:
        return None
    parts = text.split()
    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
    return yolo_to_xyxy(cx, cy, w, h, img_w, img_h)


def run_evaluation(model_path: str, dataset_dir: Path, conf_thresh: float,
                   iou_thresh: float, save_images: bool, output_dir: Path):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        return

    model = YOLO(model_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = sorted(p for p in dataset_dir.iterdir() if p.is_dir() and p.name.startswith("run_"))
    if not runs:
        print(f"No run_* folders in {dataset_dir}")
        return

    all_results: list[dict] = []

    for run_dir in runs:
        rgb_dir = run_dir / "rgb"
        tele_path = run_dir / "telemetry.csv"
        if not rgb_dir.exists():
            continue

        telemetry = load_telemetry(tele_path) if tele_path.exists() else {}
        rgb_files = sorted(rgb_dir.glob("frame_*.png"))

        for rgb_path in rgb_files:
            frame_name = rgb_path.stem
            frame_idx = int(frame_name.replace("frame_", ""))
            tele_row = telemetry.get(frame_idx, {})
            gt_dist = float(tele_row.get("dist_xy_m", -1))

            img = cv2.imread(str(rgb_path))
            if img is None:
                continue
            img_h, img_w = img.shape[:2]

            results = model(img, conf=conf_thresh, verbose=False)
            boxes = results[0].boxes

            detected = len(boxes) > 0
            best_conf = 0.0
            best_box = None

            if detected:
                confs = boxes.conf.cpu().numpy()
                xyxys = boxes.xyxy.cpu().numpy()
                best_idx = confs.argmax()
                best_conf = float(confs[best_idx])
                best_box = xyxys[best_idx].tolist()

            row = {
                "run": run_dir.name,
                "frame": frame_idx,
                "detected": int(detected),
                "confidence": round(best_conf, 4),
                "bbox_x1": round(best_box[0], 1) if best_box else "",
                "bbox_y1": round(best_box[1], 1) if best_box else "",
                "bbox_x2": round(best_box[2], 1) if best_box else "",
                "bbox_y2": round(best_box[3], 1) if best_box else "",
                "gt_distance_m": round(gt_dist, 2),
            }
            all_results.append(row)

            if save_images and detected:
                x1, y1, x2, y2 = [int(v) for v in best_box]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"drone {best_conf:.2f}"
                cv2.putText(img, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                if gt_dist > 0:
                    cv2.putText(img, f"{gt_dist:.0f}m", (x1, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                vis_dir = output_dir / "vis" / run_dir.name
                vis_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(vis_dir / f"{frame_name}.jpg"), img)

    # Save results CSV
    csv_path = output_dir / "detection_results.csv"
    if all_results:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            writer.writeheader()
            writer.writerows(all_results)
        print(f"Results CSV: {csv_path.resolve()}")

    # Print summary metrics
    print_summary(all_results)


def print_summary(results: list[dict]):
    if not results:
        print("No results to summarise.")
        return

    total = len(results)
    detected = sum(r["detected"] for r in results)
    print(f"\n{'='*60}")
    print(f"Total frames: {total}  |  Detected: {detected}  |  Missed: {total - detected}")
    print(f"Overall detection rate: {detected/total*100:.1f}%")

    # Detection rate by distance bands
    bands = [(0, 10), (10, 25), (25, 50), (50, 100), (100, 150), (150, 250)]
    print(f"\n{'Distance band':>15s}  {'Frames':>7s}  {'Detected':>9s}  {'Rate':>6s}  {'Avg Conf':>9s}")
    print("-" * 55)
    for lo, hi in bands:
        in_band = [r for r in results if lo <= r["gt_distance_m"] < hi]
        if not in_band:
            continue
        det = sum(r["detected"] for r in in_band)
        avg_conf = np.mean([r["confidence"] for r in in_band if r["detected"]]) if det else 0
        rate = det / len(in_band) * 100
        print(f"  {lo:>3d}–{hi:<3d} m      {len(in_band):>5d}    {det:>5d}     {rate:>5.1f}%    {avg_conf:>6.3f}")

    print(f"{'='*60}")


def main():
    p = argparse.ArgumentParser(description="Run YOLOv8 inference and evaluate drone detection")
    p.add_argument("--model", type=str, required=True,
                   help="Path to trained YOLO model (e.g. runs/train/drone_detector/weights/best.pt)")
    p.add_argument("--dataset", type=str, default="dataset",
                   help="Path to dataset root (contains run_* folders)")
    p.add_argument("--conf", type=float, default=0.25,
                   help="Confidence threshold (default: 0.25)")
    p.add_argument("--iou-thresh", type=float, default=0.5,
                   help="IoU threshold for TP matching (default: 0.5)")
    p.add_argument("--save-images", action="store_true",
                   help="Save annotated images with detections")
    p.add_argument("--output", type=str, default="runs/detect",
                   help="Output directory for results")
    args = p.parse_args()

    run_evaluation(
        model_path=args.model,
        dataset_dir=Path(args.dataset),
        conf_thresh=args.conf,
        iou_thresh=args.iou_thresh,
        save_images=args.save_images,
        output_dir=Path(args.output),
    )


if __name__ == "__main__":
    main()
