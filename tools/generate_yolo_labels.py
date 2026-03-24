"""
generate_yolo_labels.py — Auto-generate YOLO bounding-box labels from AirSim
segmentation masks and prepare the dataset_yolo/ folder structure for training.

Usage:
    python tools/generate_yolo_labels.py --dataset dataset --output dataset_yolo --val-ratio 0.2
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

# Drone segmentation color discovered by inspecting seg/ PNGs (BGR order).
DRONE_BGR = np.array([95, 223, 159], dtype=np.uint8)

# Seg images are 640x480, RGB images are 1280x960 → 2x scale.
SEG_W, SEG_H = 640, 480
RGB_W, RGB_H = 1280, 960
SCALE_X = RGB_W / SEG_W
SCALE_Y = RGB_H / SEG_H

MIN_PIXELS = 2  # ignore masks with fewer drone pixels than this


def extract_bbox(seg_path: Path) -> tuple[float, float, float, float] | None:
    """Return YOLO-format (cx, cy, w, h) normalised to RGB resolution, or None."""
    img = cv2.imread(str(seg_path))
    if img is None:
        return None

    drone_mask = np.all(img == DRONE_BGR, axis=-1)
    if drone_mask.sum() < MIN_PIXELS:
        return None

    ys, xs = np.where(drone_mask)
    x1 = float(xs.min()) * SCALE_X
    x2 = float(xs.max() + 1) * SCALE_X
    y1 = float(ys.min()) * SCALE_Y
    y2 = float(ys.max() + 1) * SCALE_Y

    cx = (x1 + x2) / 2 / RGB_W
    cy = (y1 + y2) / 2 / RGB_H
    w = (x2 - x1) / RGB_W
    h = (y2 - y1) / RGB_H
    return cx, cy, w, h


def load_distances(telemetry_csv: Path) -> dict[int, float]:
    """Map frame index → distance_xy_m from telemetry."""
    distances: dict[int, float] = {}
    with open(telemetry_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            distances[int(row["frame"])] = float(row["dist_xy_m"])
    return distances


def process_dataset(dataset_dir: Path, output_dir: Path, val_ratio: float, seed: int):
    random.seed(seed)

    runs = sorted(p for p in dataset_dir.iterdir() if p.is_dir() and p.name.startswith("run_"))
    if not runs:
        print(f"No run_* folders found in {dataset_dir}")
        return

    # Collect all valid (image_path, label_str, distance) tuples.
    samples: list[tuple[Path, str, float]] = []
    skipped = 0

    for run_dir in runs:
        seg_dir = run_dir / "seg"
        rgb_dir = run_dir / "rgb"
        tele_csv = run_dir / "telemetry.csv"

        if not seg_dir.exists() or not rgb_dir.exists():
            continue

        distances = load_distances(tele_csv) if tele_csv.exists() else {}

        for seg_file in sorted(seg_dir.glob("frame_*.png")):
            frame_name = seg_file.stem
            rgb_file = rgb_dir / f"{frame_name}.png"
            if not rgb_file.exists():
                continue

            bbox = extract_bbox(seg_file)
            if bbox is None:
                skipped += 1
                continue

            cx, cy, w, h = bbox
            label_line = f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
            frame_idx = int(frame_name.replace("frame_", ""))
            dist = distances.get(frame_idx, -1.0)
            samples.append((rgb_file, label_line, dist))

    print(f"Valid samples: {len(samples)}  |  Skipped (no drone visible): {skipped}")
    if not samples:
        return

    random.shuffle(samples)
    split_idx = int(len(samples) * (1 - val_ratio))
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    for split_name, split_samples in [("train", train_samples), ("val", val_samples)]:
        img_dir = output_dir / "images" / split_name
        lbl_dir = output_dir / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for i, (rgb_path, label_line, _dist) in enumerate(split_samples):
            # Use unique name: run + frame to avoid collisions
            run_name = rgb_path.parent.parent.name
            fname = f"{run_name}_{rgb_path.stem}"
            shutil.copy2(rgb_path, img_dir / f"{fname}.png")
            (lbl_dir / f"{fname}.txt").write_text(label_line + "\n", encoding="utf-8")

    # dataset.yaml (use absolute path for reliability)
    yaml_path = output_dir / "dataset.yaml"
    abs_out = output_dir.resolve().as_posix()
    yaml_path.write_text(
        f"path: {abs_out}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: 1\n"
        f"names: ['drone']\n",
        encoding="utf-8",
    )

    print(f"Train: {len(train_samples)}  |  Val: {len(val_samples)}")
    print(f"Output: {output_dir.resolve()}")
    print(f"YAML:   {yaml_path.resolve()}")

    # Distance distribution summary
    all_dists = [d for _, _, d in samples if d > 0]
    if all_dists:
        print(f"Distance range: {min(all_dists):.1f} – {max(all_dists):.1f} m  "
              f"(mean {np.mean(all_dists):.1f} m)")


def main():
    p = argparse.ArgumentParser(description="Generate YOLO labels from AirSim segmentation masks")
    p.add_argument("--dataset", type=str, default="dataset",
                   help="Path to the dataset root (contains run_* folders)")
    p.add_argument("--output", type=str, default="dataset_yolo",
                   help="Output directory for YOLO dataset")
    p.add_argument("--val-ratio", type=float, default=0.2,
                   help="Fraction of data for validation (default: 0.2)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for train/val split")
    args = p.parse_args()

    process_dataset(
        dataset_dir=Path(args.dataset),
        output_dir=Path(args.output),
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
