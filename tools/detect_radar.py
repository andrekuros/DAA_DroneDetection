"""
detect_radar.py — Summarise Echo (radar) detections vs ground-truth telemetry.

Radar frames are saved as N×6 arrays:
    x, y, z, attenuation, distance, reflections

We treat any non-empty frame as a positive radar detection and use the
minimum `distance` value in the frame as the estimated range to the
closest target (drone + clutter).

Usage:
    python tools/detect_radar.py --dataset dataset
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def load_telemetry(tele_path: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    with open(tele_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[int(row["frame"])] = row
    return rows


def detect_in_radar_frame(points: np.ndarray) -> tuple[bool, float, int]:
    """
    Simple radar detection heuristic.

    Returns (detected, est_range_m, n_returns).
    """
    if points.size == 0:
        return False, 0.0, 0

    # points: [x,y,z,attenuation,distance,reflections]
    if points.shape[1] < 5:
        return False, 0.0, 0

    dists = points[:, 4]
    # Ignore obviously invalid zeros
    valid = dists > 0.1
    if not np.any(valid):
        return False, 0.0, 0

    dists = dists[valid]
    est_range = float(dists.min())
    n_returns = int(valid.sum())
    return True, est_range, n_returns


def run_evaluation(dataset_dir: Path, output_dir: Path) -> None:
    runs = sorted(
        p for p in dataset_dir.iterdir()
        if p.is_dir() and p.name.startswith("run_")
    )
    if not runs:
        print(f"No run_* folders in {dataset_dir}")
        return

    all_results: list[dict] = []

    for run_dir in runs:
        radar_dir = run_dir / "radar"
        tele_path = run_dir / "telemetry.csv"

        if not radar_dir.exists():
            print(f"  Skipping {run_dir.name} (no radar/ folder)")
            continue

        telemetry = load_telemetry(tele_path) if tele_path.exists() else {}
        npy_files = sorted(radar_dir.glob("frame_*.npy"))
        if not npy_files:
            continue

        det_count = 0
        for npy_path in npy_files:
            frame_idx = int(npy_path.stem.replace("frame_", ""))
            tele_row = telemetry.get(frame_idx, {})
            gt_dist = float(tele_row.get("dist_xy_m", -1))

            points = np.load(npy_path)
            detected, est_range, n_returns = detect_in_radar_frame(points)

            range_error = abs(est_range - gt_dist) if (detected and gt_dist > 0) else -1

            row = {
                "run": run_dir.name,
                "frame": frame_idx,
                "detected": int(detected),
                "est_range_m": round(est_range, 2),
                "n_returns": n_returns,
                "gt_distance_m": round(gt_dist, 2),
                "range_error_m": round(range_error, 2) if range_error >= 0 else "",
            }
            all_results.append(row)
            det_count += int(detected)

        print(f"  {run_dir.name}: {det_count}/{len(npy_files)} frames detected (radar)")

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "radar_results.csv"
    if all_results:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nResults CSV: {csv_path.resolve()}")

    print_summary(all_results)


def print_summary(results: list[dict]) -> None:
    if not results:
        print("No results to summarise.")
        return

    total = len(results)
    detected = sum(r["detected"] for r in results)
    print(f"\n{'='*65}")
    print("Radar sensor: CSM Echo (RadarCSM)")
    print(f"Total frames: {total}  |  Detected: {detected}  |  Missed: {total - detected}")
    print(f"Overall detection rate: {detected/total*100:.1f}%")

    errors = [
        r["range_error_m"] for r in results
        if r["detected"] and r["range_error_m"] != ""
    ]
    if errors:
        errs = np.array(errors, dtype=float)
        print("\nRange estimation (detected frames):")
        print(f"  Mean abs error: {errs.mean():.1f} m")
        print(f"  Median error:   {np.median(errs):.1f} m")
        print(f"  Std:            {errs.std():.1f} m")

    bands = [(0, 10), (10, 25), (25, 50), (50, 100), (100, 150), (150, 250)]
    print(f"\n{'Distance band':>15s}  {'Frames':>7s}  {'Detected':>9s}  {'Rate':>6s}  {'Avg Ret':>8s}  {'Avg RngErr':>11s}")
    print("-" * 65)
    for lo, hi in bands:
        in_band = [r for r in results if lo <= r["gt_distance_m"] < hi]
        if not in_band:
            continue
        det = sum(r["detected"] for r in in_band)
        avg_ret = np.mean([r["n_returns"] for r in in_band if r["detected"]]) if det else 0
        errs = [
            r["range_error_m"] for r in in_band
            if r["detected"] and r["range_error_m"] != ""
        ]
        avg_err = np.mean(errs) if errs else 0
        rate = det / len(in_band) * 100
        print(f"  {lo:>3d}–{hi:<3d} m      {len(in_band):>5d}    {det:>5d}     {rate:>5.1f}%   {avg_ret:>6.1f}    {avg_err:>8.1f} m")

    print(f"{'='*65}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Summarise Echo radar detections vs ground truth"
    )
    p.add_argument(
        "--dataset", type=str, default="dataset",
        help="Path to dataset root (contains run_* folders)",
    )
    p.add_argument(
        "--output", type=str, default="runs/detect",
        help="Output directory for results",
    )
    args = p.parse_args()

    run_evaluation(
        dataset_dir=Path(args.dataset),
        output_dir=Path(args.output),
    )


if __name__ == "__main__":
    main()

