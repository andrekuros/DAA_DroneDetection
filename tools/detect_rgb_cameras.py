"""
detect_rgb_cameras.py — YOLO inference on multi-camera YOLO folders under data/.

Expected layout (per camera):
  data/<camera_name>/{train,valid,test}/images/*.png
  data/<camera_name>/{train,valid,test}/labels/*.txt

Image stem: run_<id>_<scenario>_frame_<NNNNNN>  (split on '_frame_').

Optional telemetry (--sim-dataset) for distance-banded metrics:
  <sim-dataset>/<run_name>/telemetry.csv

Outputs (default runs/detect):
  detection_rgb_<camera>.csv
  detection_rgb_all_cameras.csv  (concatenated)

Metrics:
  - detected: any prediction above conf
  - tp: IoU(pred, GT label) >= --iou-thresh (GT must exist)
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import cv2
import numpy as np


def load_telemetry_map(sim_dataset: Path) -> dict[tuple[str, int], dict]:
    """Map (run_name, frame_idx) -> telemetry row."""
    out: dict[tuple[str, int], dict] = {}
    if not sim_dataset.is_dir():
        return out
    for run_dir in sorted(sim_dataset.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
            continue
        tele = run_dir / "telemetry.csv"
        if not tele.exists():
            continue
        with open(tele, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                fi = int(row["frame"])
                out[(run_dir.name, fi)] = row
    return out


def parse_run_frame(stem: str) -> tuple[str, int] | None:
    if "_frame_" not in stem:
        return None
    run, rest = stem.rsplit("_frame_", 1)
    if not run.startswith("run_") or not rest.isdigit():
        return None
    return run, int(rest)


def yolo_to_xyxy(cx, cy, w, h, img_w, img_h):
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return [x1, y1, x2, y2]


def load_gt_label(label_path: Path, img_w: int, img_h: int):
    if not label_path.exists():
        return None
    text = label_path.read_text().strip()
    if not text:
        return None
    parts = text.split()
    if len(parts) < 5:
        return None
    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
    return yolo_to_xyxy(cx, cy, w, h, img_w, img_h)


def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def is_camera_dir(p: Path) -> bool:
    if not p.is_dir():
        return False
    if p.name in ("train", "valid", "test"):
        return False
    return (p / "train" / "images").is_dir()


def evaluate_camera(
    camera: str,
    camera_root: Path,
    model,
    tele_map: dict[tuple[str, int], dict],
    conf_thresh: float,
    iou_thresh: float,
    save_images: bool,
    out_vis: Path,
) -> list[dict]:
    rows: list[dict] = []
    splits = ["train", "valid", "test"]
    for split in splits:
        img_dir = camera_root / split / "images"
        if not img_dir.is_dir():
            continue
        for img_path in sorted(img_dir.glob("*.png")):
            parsed = parse_run_frame(img_path.stem)
            if parsed is None:
                continue
            run_name, frame_idx = parsed
            label_path = camera_root / split / "labels" / f"{img_path.stem}.txt"

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            gt_box = load_gt_label(label_path, w, h)

            tele_row = tele_map.get((run_name, frame_idx), {})
            try:
                gt_dist = float(tele_row.get("dist_xy_m", -1) or -1)
            except (TypeError, ValueError):
                gt_dist = -1.0

            results = model(img, conf=conf_thresh, verbose=False)
            boxes = results[0].boxes

            detected = len(boxes) > 0
            best_conf = 0.0
            best_box = None
            best_iou = 0.0

            if detected:
                confs = boxes.conf.cpu().numpy()
                xyxys = boxes.xyxy.cpu().numpy()
                if gt_box is not None:
                    best_idx = 0
                    for i in range(len(confs)):
                        iou_v = iou(xyxys[i].tolist(), gt_box)
                        if iou_v > best_iou:
                            best_iou = iou_v
                            best_idx = i
                    best_conf = float(confs[best_idx])
                    best_box = xyxys[best_idx].tolist()
                else:
                    best_idx = int(confs.argmax())
                    best_conf = float(confs[best_idx])
                    best_box = xyxys[best_idx].tolist()

            tp = int(detected and gt_box is not None and best_iou >= iou_thresh)

            row = {
                "camera": camera,
                "split": split,
                "run": run_name,
                "frame": frame_idx,
                "detected": int(detected),
                "tp": tp,
                "iou": round(best_iou, 4) if gt_box is not None else "",
                "confidence": round(best_conf, 4) if detected else 0.0,
                "has_gt_label": int(gt_box is not None),
                "gt_distance_m": round(gt_dist, 2) if gt_dist > 0 else "",
            }
            rows.append(row)

            if save_images and detected and best_box:
                vis = img.copy()
                x1, y1, x2, y2 = [int(v) for v in best_box]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    vis,
                    f"{best_conf:.2f} IoU={best_iou:.2f}",
                    (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
                d = out_vis / camera / split
                d.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(d / f"{img_path.stem}.jpg"), vis)

    return rows


def print_summary(rows: list[dict], title: str) -> None:
    if not rows:
        print(f"No results for {title}")
        return
    n = len(rows)
    det = sum(r["detected"] for r in rows)
    with_gt = [r for r in rows if r["has_gt_label"]]
    tp = sum(r["tp"] for r in with_gt)
    print(f"\n{'='*60}\n{title}\n{'='*60}")
    print(f"Frames: {n}  |  Any detection: {det} ({det/n*100:.1f}%)")
    if with_gt:
        print(f"With GT label: {len(with_gt)}  |  TP (IoU): {tp} ({tp/len(with_gt)*100:.1f}%)")

    bands = [(0, 10), (10, 25), (25, 50), (50, 100), (100, 150), (150, 250)]
    banded = [r for r in rows if r["gt_distance_m"] != ""]
    if banded:
        print(f"\n{'Distance band':>15s}  {'Frames':>7s}  {'Det%':>7s}  {'TP%':>7s}")
        print("-" * 45)
        for lo, hi in bands:
            ib = [r for r in banded if lo <= float(r["gt_distance_m"]) < hi]
            if not ib:
                continue
            d = sum(r["detected"] for r in ib)
            t = sum(r["tp"] for r in ib)
            print(
                f"  {lo:>3d}-{hi:<3d} m   {len(ib):>5d}   {d/len(ib)*100:6.1f}%  {t/len(ib)*100:6.1f}%"
            )
    print(f"{'='*60}\n")


def main() -> None:
    p = argparse.ArgumentParser(description="YOLO eval on multi-camera data/ folders")
    p.add_argument("--data-root", type=str, default="data", help="Root with camera subfolders")
    p.add_argument(
        "--sim-dataset",
        type=str,
        default="dataset",
        help="Optional Cosys-AirSim run folders for telemetry (dist_xy_m)",
    )
    p.add_argument("--model", type=str, required=True, help="YOLO weights .pt")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou-thresh", type=float, default=0.5, help="TP if IoU >= this vs GT label")
    p.add_argument("--output", type=str, default="runs/detect", help="Output directory")
    p.add_argument("--save-images", action="store_true")
    p.add_argument(
        "--cameras",
        type=str,
        default="",
        help="Comma-separated camera folder names; empty = all under data-root",
    )
    args = p.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: pip install ultralytics")
        return

    root = Path(args.data_root)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    sim_path = Path(args.sim_dataset)
    tele_map = load_telemetry_map(sim_path)
    if tele_map:
        print(f"Loaded telemetry for {len(set(k[0] for k in tele_map))} runs from {sim_path}")
    else:
        print(f"No telemetry loaded (missing or empty: {sim_path}) — distance bands skipped.")

    model = YOLO(args.model)

    if args.cameras.strip():
        cams = [c.strip() for c in args.cameras.split(",") if c.strip()]
    else:
        cams = sorted(d.name for d in root.iterdir() if is_camera_dir(d))

    if not cams:
        print(f"No camera folders found under {root}")
        return

    all_rows: list[dict] = []
    for cam in cams:
        cam_root = root / cam
        if not cam_root.is_dir():
            print(f"Skip missing camera: {cam}")
            continue
        rows = evaluate_camera(
            cam,
            cam_root,
            model,
            tele_map,
            args.conf,
            args.iou_thresh,
            args.save_images,
            out_dir / "vis_rgb_cameras",
        )
        all_rows.extend(rows)
        print_summary(rows, f"Camera: {cam}")
        if rows:
            csv_path = out_dir / f"detection_rgb_{cam}.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            print(f"Wrote {csv_path}")

    if all_rows:
        merged = out_dir / "detection_rgb_all_cameras.csv"
        with open(merged, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        print(f"Merged: {merged}")
        print_summary(all_rows, "ALL CAMERAS")


if __name__ == "__main__":
    main()
