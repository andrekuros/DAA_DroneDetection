"""
vis_sensors.py — Visualise RGB, Lidar and Radar detections for a given frame.

RGB:
  - Loads dataset/<run>/rgb/frame_XXXXXX.png
  - Optionally overlays YOLO bounding box from runs/detect/detection_results.csv

Lidar (os128, vlp16):
  - Loads dataset/<run>/<lidar_subdir>/frame_XXXXXX.npy (N×3 XYZ)
  - Top-down view (X forward, Y lateral) projected into a 512×512 PNG

Radar:
  - Loads dataset/<run>/radar/frame_XXXXXX.npy (N×6: x,y,z,att,dist,refl)
  - Plots distance vs bearing (atan2(y, x)) into a 512×512 PNG

Usage examples:
    python tools/vis_sensors.py --run run_001_az000_day --frame 40 --sensor rgb
    python tools/vis_sensors.py --run run_001_az000_day --frame 40 --sensor os128
    python tools/vis_sensors.py --run run_001_az000_day --frame 40 --sensor radar
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_rgb_with_bbox(run: str, frame: int) -> np.ndarray:
    img_path = ROOT / "dataset" / run / "rgb" / f"frame_{frame:06d}.png"
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"RGB image not found: {img_path}")

    det_csv = ROOT / "runs" / "detect" / "detection_results.csv"
    if det_csv.exists():
        with open(det_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["run"] == run and int(row["frame"]) == frame and int(row["detected"]):
                    x1 = int(float(row["bbox_x1"]))
                    y1 = int(float(row["bbox_y1"]))
                    x2 = int(float(row["bbox_x2"]))
                    y2 = int(float(row["bbox_y2"]))
                    conf = float(row["confidence"])
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        img,
                        f"drone {conf:.2f}",
                        (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )
                    break
    return img


def render_lidar_topdown(points: np.ndarray, max_range: float = 120.0) -> np.ndarray:
    """
    Render a simple top-down view (X forward, Y lateral) into 512×512.
    """
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    if points.size == 0:
        return img

    mask = ~np.all(points == 0, axis=1)
    pts = points[mask]
    if pts.shape[0] == 0:
        return img

    x = np.clip(pts[:, 0], -max_range, max_range)
    y = np.clip(pts[:, 1], -max_range, max_range)

    # Map x in [0, max_range] to vertical 511→0 (sensor at bottom)
    # Map y in [-max_range, max_range] to horizontal 0→511
    xs = (1.0 - np.clip(x / max_range, 0.0, 1.0)) * 511.0
    ys = (0.5 + y / (2 * max_range)) * 511.0

    for xi, yi in zip(xs.astype(int), ys.astype(int)):
        if 0 <= xi < 512 and 0 <= yi < 512:
            img[xi, yi] = (255, 255, 255)

    # Draw sensor position
    cv2.circle(img, (256, 511), 4, (0, 0, 255), -1)
    return img


def render_radar_polar(points: np.ndarray, max_range: float = 250.0) -> np.ndarray:
    """
    Render radar points as bearing vs range in 512×512.
    Bearing mapped horizontally, range vertically.
    """
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    if points.size == 0 or points.shape[1] < 5:
        return img

    x = points[:, 0]
    y = points[:, 1]
    dist = points[:, 4]

    # Filter invalid distances
    valid = dist > 0.1
    x = x[valid]
    y = y[valid]
    dist = np.clip(dist[valid], 0.0, max_range)
    if dist.size == 0:
        return img

    bearing = np.arctan2(y, x)  # -pi..pi

    # Map bearing -pi..pi to 0..511, range 0..max_range to 511..0
    xs = ((bearing + np.pi) / (2 * np.pi)) * 511.0
    ys = (1.0 - (dist / max_range)) * 511.0

    for xi, yi in zip(xs.astype(int), ys.astype(int)):
        if 0 <= xi < 512 and 0 <= yi < 512:
            img[yi, xi] = (0, 255, 255)

    # Draw center line (bearing=0)
    center_x = int(0.5 * 511)
    cv2.line(img, (center_x, 0), (center_x, 511), (128, 128, 128), 1)
    return img


def main() -> None:
    p = argparse.ArgumentParser(description="Visualise RGB, Lidar, and Radar for a given frame.")
    p.add_argument("--run", type=str, required=True, help="Run folder name, e.g. run_001_az000_day")
    p.add_argument("--frame", type=int, required=True, help="Frame index (integer)")
    p.add_argument(
        "--sensor",
        type=str,
        required=True,
        choices=["rgb", "os128", "vlp16", "radar"],
        help="Which sensor to visualise",
    )
    p.add_argument(
        "--output",
        type=str,
        default="runs/vis",
        help="Output directory root",
    )
    args = p.parse_args()

    run = args.run
    frame = args.frame
    out_root = ROOT / args.output
    out_dir = out_root / run
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.sensor == "rgb":
        img = load_rgb_with_bbox(run, frame)
    elif args.sensor in {"os128", "vlp16"}:
        sub = "lidar_os128" if args.sensor == "os128" else "lidar_vlp16"
        npy_path = ROOT / "dataset" / run / sub / f"frame_{frame:06d}.npy"
        pts = np.load(npy_path)
        img = render_lidar_topdown(pts, max_range=120.0 if args.sensor == "os128" else 100.0)
    else:  # radar
        npy_path = ROOT / "dataset" / run / "radar" / f"frame_{frame:06d}.npy"
        pts = np.load(npy_path)
        img = render_radar_polar(pts, max_range=250.0)

    out_path = out_dir / f"{args.sensor}_frame_{frame:06d}.png"
    cv2.imwrite(str(out_path), img)
    print(f"Saved visualisation to {out_path}")


if __name__ == "__main__":
    main()

