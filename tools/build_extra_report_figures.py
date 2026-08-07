"""
Generate additional report figures for 13_multisensor_benchmark.tex:
  1. First-detection range bar chart per sensor (fig:first-detection)
  2. Multi-camera EO detection examples composite — same frame, 4 lenses with boxes
  3. Simulation encounter conditions composite — day/dusk/rain/night/snow/fog samples
  4. Val prediction grid copied from Ultralytics output

Usage:
    python tools/build_extra_report_figures.py
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
MULTICAM = ROOT / "dataset_exp30_multicam"
YOLO_ROOTS = {
    "front_center": ROOT / "dataset_exp30_multicam_yolo",
    "front_lowcost_640": ROOT / "dataset_exp30_yolo_lowcost640",
    "front_medium": ROOT / "dataset_exp30_yolo_medium",
    "front_narrow_hd": ROOT / "dataset_exp30_yolo_narrow_hd",
}
CAMERA_LABELS = {
    "front_center": "Standard Forward",
    "front_lowcost_640": "Low-Cost Wide (640 px)",
    "front_medium": "Medium FOV",
    "front_narrow_hd": "Narrow HD (Long-Range)",
}


def build_first_detection_chart(out: Path) -> None:
    """Horizontal bar chart: first-detection range per sensor with alert-level annotations."""
    import matplotlib.pyplot as plt

    sensors = ["Radar (FMCW)", "LiDAR HD (OS128)", "EO / Vision (YOLOv8s)", "LiDAR Sparse (VLP16)"]
    means = [116.9, 113.2, 79.3, 24.6]
    colors = ["#E74C3C", "#3498DB", "#2ECC71", "#9B59B6"]

    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=150)
    bars = ax.barh(sensors, means, color=colors, height=0.55, edgecolor="white", linewidth=0.8)
    ax.set_xlabel("Mean first-detection range (m)", fontsize=11)
    ax.set_title("First Detection Range by Sensor Modality — 29 Encounter Runs", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 160)
    ax.invert_yaxis()

    for bar, val in zip(bars, means):
        ax.text(val + 2, bar.get_y() + bar.get_height() / 2, f"{val:.1f} m",
                va="center", fontsize=10, fontweight="bold")

    for xv, label, ls in [(117, "Preventive\nenvelope", "--"), (73, "Warning\nenvelope", ":")]:
        ax.axvline(xv, color="gray", linestyle=ls, alpha=0.5, linewidth=1)
        ax.text(xv, -0.05, label, ha="center", va="top", fontsize=7.5, color="gray",
                transform=ax.get_xaxis_transform())

    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out.stem}")


def _draw_yolo_boxes(img: np.ndarray, label_path: Path) -> np.ndarray:
    """Draw YOLO boxes from a label .txt onto an image."""
    out = img.copy()
    H, W = out.shape[:2]
    if not label_path.exists():
        return out
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        _, cx, cy, bw, bh = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        x1 = int((cx - bw / 2) * W)
        y1 = int((cy - bh / 2) * H)
        x2 = int((cx + bw / 2) * W)
        y2 = int((cy + bh / 2) * H)
        thick = max(2, min(W, H) // 300)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), thick, lineType=cv2.LINE_AA)
        cv2.putText(out, "drone", (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
    return out


def _find_matching_frame_across_cameras(run_tag: str, frame_tag: str) -> dict[str, tuple[Path, Path]]:
    """For a given run+frame, find the image and label across all camera YOLO datasets."""
    result: dict[str, tuple[Path, Path]] = {}
    for cam, yolo_root in YOLO_ROOTS.items():
        for split in ("train", "val"):
            img_dir = yolo_root / "images" / split
            lbl_dir = yolo_root / "labels" / split
            if not img_dir.exists():
                continue
            for img_path in img_dir.glob("*" + run_tag + "*" + frame_tag + "*.png"):
                lbl_path = lbl_dir / (img_path.stem + ".txt")
                if lbl_path.exists():
                    result[cam] = (img_path, lbl_path)
                    break
        if cam in result:
            continue
    return result


def build_multicam_detection_composite(out: Path) -> None:
    """2x2 grid: same encounter frame shown through 4 different camera lenses with detection boxes."""
    random.seed(42)

    multicam_yolo = ROOT / "dataset_exp30_multicam_yolo"
    img_dir = multicam_yolo / "images" / "val"
    lbl_dir = multicam_yolo / "labels" / "val"
    if not img_dir.exists():
        print("  [SKIP] multicam_yolo val not found")
        return

    candidates: dict[str, list[str]] = {}
    for lbl in sorted(lbl_dir.glob("*.txt")):
        parts = lbl.stem.split("_")
        for i, p in enumerate(parts):
            if p.startswith("run"):
                run_tag = "_".join(parts[i:i+2])
                frame_tag = parts[-1]
                if run_tag not in candidates:
                    candidates[run_tag] = []
                candidates[run_tag].append(frame_tag)
                break

    best_match = None
    for run_tag, frames in sorted(candidates.items()):
        for frame_tag in frames:
            result = _find_matching_frame_across_cameras(run_tag, frame_tag)
            if len(result) >= 4 and best_match is None:
                best_match = result
                break
        if best_match:
            break

    if not best_match or len(best_match) < 3:
        best_match = {}
        for cam, yolo_root in YOLO_ROOTS.items():
            for split in ("val", "train"):
                lbl_dir_c = yolo_root / "labels" / split
                img_dir_c = yolo_root / "images" / split
                if not lbl_dir_c.exists():
                    continue
                lbls = sorted(lbl_dir_c.glob("*day*.txt"))
                if not lbls:
                    lbls = sorted(lbl_dir_c.glob("*.txt"))
                if lbls:
                    lp = lbls[min(3, len(lbls) - 1)]
                    ip = img_dir_c / (lp.stem + ".png")
                    if ip.exists():
                        best_match[cam] = (ip, lp)
                        break

    order = ["front_center", "front_narrow_hd", "front_medium", "front_lowcost_640"]
    panels = []
    for cam in order:
        if cam not in best_match:
            continue
        ip, lp = best_match[cam]
        img = cv2.imread(str(ip))
        if img is None:
            continue
        img = _draw_yolo_boxes(img, lp)
        h, w = img.shape[:2]
        target_w = 640
        scale = target_w / w
        img = cv2.resize(img, (target_w, int(h * scale)), interpolation=cv2.INTER_AREA)
        label = CAMERA_LABELS.get(cam, cam)
        cv2.putText(img, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 5, cv2.LINE_AA)
        cv2.putText(img, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 180, 255), 2, cv2.LINE_AA)
        panels.append(img)

    if len(panels) < 2:
        print("  [SKIP] not enough panels for composite")
        return

    max_h = max(p.shape[0] for p in panels)
    padded = []
    for p in panels:
        if p.shape[0] < max_h:
            pad = np.zeros((max_h - p.shape[0], p.shape[1], 3), dtype=np.uint8)
            p = np.vstack([p, pad])
        padded.append(p)

    n = len(padded)
    top = np.hstack(padded[: n // 2]) if n >= 2 else padded[0]
    bot = np.hstack(padded[n // 2:]) if n > 2 else np.zeros_like(top)
    if top.shape[1] != bot.shape[1]:
        target_w2 = max(top.shape[1], bot.shape[1])
        if top.shape[1] < target_w2:
            top = np.hstack([top, np.zeros((top.shape[0], target_w2 - top.shape[1], 3), dtype=np.uint8)])
        if bot.shape[1] < target_w2:
            bot = np.hstack([bot, np.zeros((bot.shape[0], target_w2 - bot.shape[1], 3), dtype=np.uint8)])
    composite = np.vstack([top, bot])

    cv2.imwrite(str(out.with_suffix(".png")), composite, [cv2.IMWRITE_PNG_COMPRESSION, 5])
    print(f"  -> {out.stem} ({composite.shape[1]}x{composite.shape[0]})")


def build_encounter_conditions_composite(out: Path) -> None:
    """3x2 grid showing encounters under different weather/lighting conditions."""
    condition_keywords = [
        ("day", "Clear day"),
        ("dusk", "Dusk"),
        ("rain", "Rain"),
        ("night", "Night"),
        ("snow", "Snow"),
        ("dust", "Dust/Haze"),
    ]
    cam = "front_center"
    cam_dir = MULTICAM / cam
    if not cam_dir.exists():
        cam_dir = MULTICAM / "front_medium"
    if not cam_dir.exists():
        print("  [SKIP] no camera folder found for conditions composite")
        return

    panels = []
    for keyword, label in condition_keywords:
        runs = sorted(cam_dir.glob(f"run_*{keyword}*"))
        if not runs:
            runs = sorted(cam_dir.glob("run_*"))
        found = False
        for run_dir in runs:
            rgb_dir = run_dir / "rgb"
            if not rgb_dir.exists():
                continue
            frames = sorted(rgb_dir.glob("frame_*.png"))
            mid = len(frames) // 3
            if mid >= len(frames):
                mid = 0
            if not frames:
                continue
            img = cv2.imread(str(frames[mid]))
            if img is None:
                continue
            h, w = img.shape[:2]
            target_w = 480
            scale = target_w / w
            img = cv2.resize(img, (target_w, int(h * scale)), interpolation=cv2.INTER_AREA)
            cv2.putText(img, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(img, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            panels.append(img)
            found = True
            break
        if not found:
            blank = np.zeros((360, 480, 3), dtype=np.uint8)
            cv2.putText(blank, f"{label} (N/A)", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)
            panels.append(blank)

    max_h = max(p.shape[0] for p in panels)
    padded = []
    for p in panels:
        if p.shape[0] < max_h:
            pad = np.zeros((max_h - p.shape[0], p.shape[1], 3), dtype=np.uint8)
            p = np.vstack([p, pad])
        padded.append(p)

    rows = []
    cols = 3
    for r in range(0, len(padded), cols):
        chunk = padded[r : r + cols]
        while len(chunk) < cols:
            chunk.append(np.zeros_like(chunk[0]))
        rows.append(np.hstack(chunk))
    composite = np.vstack(rows)

    cv2.imwrite(str(out.with_suffix(".png")), composite, [cv2.IMWRITE_PNG_COMPRESSION, 5])
    print(f"  -> {out.stem} ({composite.shape[1]}x{composite.shape[0]})")


def copy_val_predictions(out: Path) -> None:
    """Copy and label val_batch prediction grids from Ultralytics output."""
    import shutil
    src = ROOT / "runs" / "detect" / "runs" / "train" / "exp30_multicam_yolov8s" / "val_batch0_pred.jpg"
    if src.exists():
        shutil.copy2(src, out.with_suffix(".jpg"))
        print(f"  -> {out.stem} (from Ultralytics val grid)")
    else:
        print(f"  [SKIP] {src} not found")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    print("Building extra report figures...")

    print("\n1. First-detection range chart")
    build_first_detection_chart(FIGURES / "exp30_first_detection_range")

    print("\n2. Multi-camera detection composite")
    build_multicam_detection_composite(FIGURES / "exp30_multicam_detection_examples")

    print("\n3. Encounter conditions composite")
    build_encounter_conditions_composite(FIGURES / "exp30_encounter_conditions")

    print("\n4. Val prediction grid")
    copy_val_predictions(FIGURES / "exp30_val_predictions")

    print(f"\nAll figures in: {FIGURES.resolve()}")


if __name__ == "__main__":
    main()
