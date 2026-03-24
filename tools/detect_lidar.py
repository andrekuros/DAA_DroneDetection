"""
detect_lidar.py — Detect drones in Lidar point clouds using DBSCAN clustering
and evaluate against ground-truth telemetry.

The algorithm:
  1. Load each frame's .npy point cloud (N×3, XYZ in sensor frame)
  2. Remove zero / near-body points (< min_dist from sensor)
  3. Run DBSCAN clustering on remaining points
  4. Score each cluster: isolated, small, airborne → drone candidate
  5. Compare estimated range with ground-truth distance from telemetry

Usage:
    python tools/detect_lidar.py --dataset dataset --sensor os128
    python tools/detect_lidar.py --dataset dataset --sensor vlp16 --min-dist 1.0
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN


SENSOR_MAP = {
    "os128": "lidar_os128",
    "vlp16": "lidar_vlp16",
}


def load_telemetry(tele_path: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    with open(tele_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[int(row["frame"])] = row
    return rows


def detect_drone_in_cloud(
    points: np.ndarray,
    min_dist: float = 1.5,
    eps: float = 2.0,
    min_samples: int = 2,
    max_cluster_pts: int = 500,
) -> tuple[bool, float, int]:
    """Detect a drone in a single Lidar frame.

    Returns (detected, estimated_range_m, n_points_in_cluster).
    """
    if points.shape[0] == 0:
        return False, 0.0, 0

    # Remove zero points
    mask = ~np.all(points == 0, axis=1)
    pts = points[mask]
    if pts.shape[0] == 0:
        return False, 0.0, 0

    # Remove near-body points (sensor housing, self-returns)
    dists = np.linalg.norm(pts, axis=1)
    far_mask = dists > min_dist
    pts = pts[far_mask]
    dists = dists[far_mask]

    if pts.shape[0] == 0:
        return False, 0.0, 0

    if pts.shape[0] == 1:
        return True, float(dists[0]), 1

    # DBSCAN clustering
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(pts)

    unique_labels = set(labels)
    unique_labels.discard(-1)  # noise

    # Also consider noise points as singleton detections
    noise_pts = pts[labels == -1]
    noise_dists = dists[labels == -1]

    best_range = 0.0
    best_score = -1.0
    best_npts = 0
    detected = False

    for lbl in unique_labels:
        cluster_mask = labels == lbl
        cluster = pts[cluster_mask]
        n_pts = cluster.shape[0]

        if n_pts > max_cluster_pts:
            continue

        centroid = cluster.mean(axis=0)
        cluster_range = float(np.linalg.norm(centroid))

        # Cluster compactness (smaller spread = more drone-like)
        spread = np.linalg.norm(cluster - centroid, axis=1).max()

        # Score: prefer compact, small clusters at range
        # A drone should be a small, tight cluster far from the sensor
        score = cluster_range / (spread + 0.5) * min(n_pts, 20)

        if score > best_score:
            best_score = score
            best_range = cluster_range
            best_npts = n_pts
            detected = True

    # If no clusters found, check for isolated noise points (single returns)
    if not detected and noise_pts.shape[0] > 0:
        best_idx = noise_dists.argmax()
        best_range = float(noise_dists[best_idx])
        best_npts = 1
        detected = True

    return detected, best_range, best_npts


def run_evaluation(
    dataset_dir: Path,
    sensor: str,
    min_dist: float,
    eps: float,
    min_samples: int,
    output_dir: Path,
):
    subfolder = SENSOR_MAP[sensor]
    runs = sorted(
        p for p in dataset_dir.iterdir()
        if p.is_dir() and p.name.startswith("run_")
    )
    if not runs:
        print(f"No run_* folders in {dataset_dir}")
        return

    all_results: list[dict] = []

    for run_dir in runs:
        lidar_dir = run_dir / subfolder
        tele_path = run_dir / "telemetry.csv"

        if not lidar_dir.exists():
            print(f"  Skipping {run_dir.name} (no {subfolder}/ folder)")
            continue

        telemetry = load_telemetry(tele_path) if tele_path.exists() else {}
        npy_files = sorted(lidar_dir.glob("frame_*.npy"))

        if not npy_files:
            continue

        det_count = 0
        for npy_path in npy_files:
            frame_idx = int(npy_path.stem.replace("frame_", ""))
            tele_row = telemetry.get(frame_idx, {})
            gt_dist = float(tele_row.get("dist_xy_m", -1))

            points = np.load(npy_path)
            detected, est_range, n_pts = detect_drone_in_cloud(
                points,
                min_dist=min_dist,
                eps=eps,
                min_samples=min_samples,
            )

            range_error = abs(est_range - gt_dist) if (detected and gt_dist > 0) else -1

            row = {
                "run": run_dir.name,
                "frame": frame_idx,
                "detected": int(detected),
                "est_range_m": round(est_range, 2),
                "n_points": n_pts,
                "gt_distance_m": round(gt_dist, 2),
                "range_error_m": round(range_error, 2) if range_error >= 0 else "",
            }
            all_results.append(row)
            det_count += int(detected)

        print(f"  {run_dir.name}: {det_count}/{len(npy_files)} frames detected ({subfolder})")

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"lidar_{sensor}_results.csv"
    if all_results:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nResults CSV: {csv_path.resolve()}")

    print_summary(all_results, sensor)


def print_summary(results: list[dict], sensor: str):
    if not results:
        print("No results to summarise.")
        return

    total = len(results)
    detected = sum(r["detected"] for r in results)
    print(f"\n{'='*65}")
    print(f"Lidar sensor: {sensor.upper()}")
    print(f"Total frames: {total}  |  Detected: {detected}  |  Missed: {total - detected}")
    print(f"Overall detection rate: {detected/total*100:.1f}%")

    # Range estimation accuracy for detected frames
    errors = [r["range_error_m"] for r in results
              if r["detected"] and r["range_error_m"] != ""]
    if errors:
        errors_arr = np.array(errors, dtype=float)
        print(f"\nRange estimation (detected frames):")
        print(f"  Mean abs error: {errors_arr.mean():.1f} m")
        print(f"  Median error:   {np.median(errors_arr):.1f} m")
        print(f"  Std:            {errors_arr.std():.1f} m")

    # Detection rate by distance bands
    bands = [(0, 10), (10, 25), (25, 50), (50, 100), (100, 150), (150, 250)]
    print(f"\n{'Distance band':>15s}  {'Frames':>7s}  {'Detected':>9s}  {'Rate':>6s}  {'Avg Pts':>8s}  {'Avg RngErr':>11s}")
    print("-" * 65)
    for lo, hi in bands:
        in_band = [r for r in results if lo <= r["gt_distance_m"] < hi]
        if not in_band:
            continue
        det = sum(r["detected"] for r in in_band)
        avg_pts = np.mean([r["n_points"] for r in in_band if r["detected"]]) if det else 0
        errs = [r["range_error_m"] for r in in_band
                if r["detected"] and r["range_error_m"] != ""]
        avg_err = np.mean(errs) if errs else 0
        rate = det / len(in_band) * 100
        print(f"  {lo:>3d}–{hi:<3d} m      {len(in_band):>5d}    {det:>5d}     {rate:>5.1f}%   {avg_pts:>6.1f}    {avg_err:>8.1f} m")

    print(f"{'='*65}")


def main():
    p = argparse.ArgumentParser(
        description="Detect drones in Lidar point clouds (DBSCAN clustering)"
    )
    p.add_argument("--dataset", type=str, default="dataset",
                   help="Path to dataset root (contains run_* folders)")
    p.add_argument("--sensor", type=str, default="os128",
                   choices=list(SENSOR_MAP.keys()),
                   help="Lidar sensor to evaluate (default: os128)")
    p.add_argument("--min-dist", type=float, default=1.5,
                   help="Min distance from sensor to consider (filters body returns)")
    p.add_argument("--eps", type=float, default=2.0,
                   help="DBSCAN eps (max distance between cluster points)")
    p.add_argument("--min-samples", type=int, default=2,
                   help="DBSCAN min_samples (min points per cluster)")
    p.add_argument("--output", type=str, default="runs/detect",
                   help="Output directory for results")
    args = p.parse_args()

    run_evaluation(
        dataset_dir=Path(args.dataset),
        sensor=args.sensor,
        min_dist=args.min_dist,
        eps=args.eps,
        min_samples=args.min_samples,
        output_dir=Path(args.output),
    )


if __name__ == "__main__":
    main()
