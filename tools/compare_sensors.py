"""
compare_sensors.py — Aggregate RGB (single or multi-camera), LiDAR, and Radar.

Aligned table (same run/frame as simulation):
  - Master key list: radar CSV if present, else lidar_os128, else primary RGB camera.
  - RGB column: legacy detection_results.csv OR detection_rgb_<primary>.csv

Multi-camera RGB-only summary:
  - Reads detection_rgb_all_cameras.csv if present (from detect_rgb_cameras.py).

Outputs:
  - Console tables (distance band, lighting, first detection)
  - runs/detect/sensor_comparison_summary.csv

Usage:
    python tools/compare_sensors.py
    python tools/compare_sensors.py --primary-rgb-camera front_narrow_hd
    python tools/compare_sensors.py --runs-dir runs/detect
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BANDS = [(0, 10), (10, 25), (25, 50), (50, 100), (100, 150), (150, 250)]


@dataclass
class FrameRow:
    run: str
    frame: int
    gt_distance_m: float
    rgb_detect: int
    lidar_os_detect: int
    lidar_vlp_detect: int
    radar_detect: int


def run_condition(run: str) -> str:
    run_l = run.lower()
    if "night" in run_l:
        return "night"
    if "dawn" in run_l:
        return "dawn"
    if "dusk" in run_l:
        return "dusk"
    return "day"


def load_csv(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def idx_by_run_frame(rows: List[dict]) -> Dict[Tuple[str, int], dict]:
    out: Dict[Tuple[str, int], dict] = {}
    for r in rows:
        key = (r["run"], int(r["frame"]))
        out[key] = r
    return out


def discover_rgb_csvs(runs_dir: Path) -> List[Path]:
    return sorted(
        p
        for p in runs_dir.glob("detection_rgb_*.csv")
        if p.name != "detection_rgb_all_cameras.csv"
    )


def pick_primary_rgb_path(
    runs_dir: Path, primary: str | None, use_legacy: bool
) -> Path | None:
    if use_legacy:
        p = runs_dir / "detection_results.csv"
        return p if p.exists() else None
    if primary:
        p = runs_dir / f"detection_rgb_{primary}.csv"
        if p.exists():
            return p
    cams = discover_rgb_csvs(runs_dir)
    return cams[0] if cams else None


def gt_for_key(
    key: Tuple[str, int],
    idx_radar: Dict,
    idx_os: Dict,
    idx_vlp: Dict,
    idx_rgb: Dict,
) -> float:
    for idx in (idx_radar, idx_os, idx_vlp, idx_rgb):
        row = idx.get(key)
        if not row:
            continue
        v = row.get("gt_distance_m", "")
        if v == "" or v is None:
            continue
        try:
            d = float(v)
            if d > 0:
                return d
        except (TypeError, ValueError):
            pass
    return -1.0


def build_frame_table(
    runs_dir: Path,
    primary_rgb_camera: str | None,
    use_legacy_rgb: bool,
) -> List[FrameRow]:
    rgb_path = pick_primary_rgb_path(runs_dir, primary_rgb_camera, use_legacy_rgb)
    rgb_rows = load_csv(rgb_path) if rgb_path else []
    os_rows = load_csv(runs_dir / "lidar_os128_results.csv")
    vlp_rows = load_csv(runs_dir / "lidar_vlp16_results.csv")
    radar_rows = load_csv(runs_dir / "radar_results.csv")

    idx_rgb = idx_by_run_frame(rgb_rows)
    idx_os = idx_by_run_frame(os_rows)
    idx_vlp = idx_by_run_frame(vlp_rows)
    idx_radar = idx_by_run_frame(radar_rows)

    if idx_radar:
        keys = sorted(idx_radar.keys())
    elif idx_os:
        keys = sorted(idx_os.keys())
    elif idx_rgb:
        keys = sorted(idx_rgb.keys())
    else:
        return []

    table: List[FrameRow] = []
    for key in keys:
        run, frame = key
        gt = gt_for_key(key, idx_radar, idx_os, idx_vlp, idx_rgb)
        if gt < 0:
            continue

        def det(idx: Dict) -> int:
            row = idx.get(key)
            if not row:
                return 0
            return int(row.get("detected", 0))

        table.append(
            FrameRow(
                run=run,
                frame=frame,
                gt_distance_m=gt,
                rgb_detect=det(idx_rgb),
                lidar_os_detect=det(idx_os),
                lidar_vlp_detect=det(idx_vlp),
                radar_detect=det(idx_radar),
            )
        )
    return table


def summarise_by_band(table: List[FrameRow], rgb_label: str = "RGB") -> None:
    print("\n" + "=" * 80)
    print("Sensor comparison by distance band (aligned run/frame)")
    print(f"RGB column = {rgb_label}")
    print("=" * 80)
    print(
        f"{'Band':>10s}  {'Frames':>7s}  "
        f"{rgb_label[:7]:>7s}  {'OS128%':>7s}  {'VLP16%':>7s}  {'Radar%':>7s}"
    )
    print("-" * 80)

    for lo, hi in BANDS:
        band_rows = [r for r in table if lo <= r.gt_distance_m < hi]
        if not band_rows:
            continue
        n = len(band_rows)

        def rate(attr: str) -> float:
            return sum(getattr(r, attr) for r in band_rows) / n * 100.0

        rgb_rate = rate("rgb_detect")
        os_rate = rate("lidar_os_detect")
        vlp_rate = rate("lidar_vlp_detect")
        radar_rate = rate("radar_detect")

        print(
            f"{lo:3d}–{hi:<3d} m  "
            f"{n:7d}  "
            f"{rgb_rate:6.1f}%  {os_rate:6.1f}%  {vlp_rate:6.1f}%  {radar_rate:6.1f}%"
        )

    print("=" * 80)


def summarise_by_condition(table: List[FrameRow], rgb_label: str = "RGB") -> None:
    print("\n" + "=" * 80)
    print("Sensor comparison by lighting condition")
    print("=" * 80)
    print(
        f"{'Condition':>10s}  {'Frames':>7s}  "
        f"{rgb_label[:7]:>7s}  {'OS128%':>7s}  {'VLP16%':>7s}  {'Radar%':>7s}"
    )
    print("-" * 80)

    for cond in ["day", "dawn", "dusk", "night"]:
        rows = [r for r in table if run_condition(r.run) == cond]
        if not rows:
            continue
        n = len(rows)
        rgb = sum(r.rgb_detect for r in rows) / n * 100.0
        os128 = sum(r.lidar_os_detect for r in rows) / n * 100.0
        vlp16 = sum(r.lidar_vlp_detect for r in rows) / n * 100.0
        radar = sum(r.radar_detect for r in rows) / n * 100.0
        print(f"{cond:>10s}  {n:7d}  {rgb:6.1f}%  {os128:6.1f}%  {vlp16:6.1f}%  {radar:6.1f}%")

    print("=" * 80)


def summarise_first_detection(table: List[FrameRow]) -> None:
    by_run: Dict[str, List[FrameRow]] = defaultdict(list)
    for r in table:
        by_run[r.run].append(r)

    winner_counts = defaultdict(float)
    runs_considered = 0
    first_detection_dist: Dict[str, List[float]] = defaultdict(list)

    for run, frames in by_run.items():
        frames_sorted = sorted(frames, key=lambda r: r.frame)

        first = {}
        for sensor, attr in [
            ("rgb", "rgb_detect"),
            ("os128", "lidar_os_detect"),
            ("vlp16", "lidar_vlp_detect"),
            ("radar", "radar_detect"),
        ]:
            hit = next((f for f in frames_sorted if getattr(f, attr)), None)
            if hit is not None:
                first[sensor] = hit

        if not first:
            continue

        runs_considered += 1
        min_frame = min(fr.frame for fr in first.values())
        winners = [s for s, fr in first.items() if fr.frame == min_frame]
        for w in winners:
            winner_counts[w] += 1 / len(winners)
            first_detection_dist[w].append(first[w].gt_distance_m)

    if runs_considered == 0:
        return

    print("\nFirst detection per run (earliest frame):")
    for sensor in ["rgb", "os128", "vlp16", "radar"]:
        c = winner_counts.get(sensor, 0.0)
        print(f"  {sensor:6s}: {c:.1f} / {runs_considered} runs")

    print("\nMean distance at first detection:")
    for sensor in ["rgb", "os128", "vlp16", "radar"]:
        d = first_detection_dist.get(sensor, [])
        if d:
            print(
                f"  {sensor:6s}: mean={np.mean(d):.1f} m, median={np.median(d):.1f} m, n={len(d)}"
            )
        else:
            print(f"  {sensor:6s}: no wins")


def export_summary_csv(table: List[FrameRow], runs_dir: Path, rgb_label: str) -> None:
    out_path = runs_dir / "sensor_comparison_summary.csv"
    rows_out = []

    for lo, hi in BANDS:
        band_rows = [r for r in table if lo <= r.gt_distance_m < hi]
        if not band_rows:
            continue
        n = len(band_rows)
        rows_out.append(
            {
                "group_type": "distance_band",
                "group_value": f"{lo}-{hi}m",
                "rgb_source": rgb_label,
                "frames": n,
                "rgb_rate_pct": round(sum(r.rgb_detect for r in band_rows) / n * 100, 2),
                "os128_rate_pct": round(sum(r.lidar_os_detect for r in band_rows) / n * 100, 2),
                "vlp16_rate_pct": round(sum(r.lidar_vlp_detect for r in band_rows) / n * 100, 2),
                "radar_rate_pct": round(sum(r.radar_detect for r in band_rows) / n * 100, 2),
            }
        )

    for cond in ["day", "dawn", "dusk", "night"]:
        rows_cond = [r for r in table if run_condition(r.run) == cond]
        if not rows_cond:
            continue
        n = len(rows_cond)
        rows_out.append(
            {
                "group_type": "condition",
                "group_value": cond,
                "rgb_source": rgb_label,
                "frames": n,
                "rgb_rate_pct": round(sum(r.rgb_detect for r in rows_cond) / n * 100, 2),
                "os128_rate_pct": round(sum(r.lidar_os_detect for r in rows_cond) / n * 100, 2),
                "vlp16_rate_pct": round(sum(r.lidar_vlp_detect for r in rows_cond) / n * 100, 2),
                "radar_rate_pct": round(sum(r.radar_detect for r in rows_cond) / n * 100, 2),
            }
        )

    if rows_out:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            w.writeheader()
            w.writerows(rows_out)
        print(f"\nSaved summary CSV: {out_path}")


def summarise_multi_camera_rgb(runs_dir: Path) -> None:
    path = runs_dir / "detection_rgb_all_cameras.csv"
    rows = load_csv(path)
    if not rows:
        print("\n(No detection_rgb_all_cameras.csv — run tools/detect_rgb_cameras.py)")
        return

    print("\n" + "=" * 80)
    print("RGB multi-camera summary (all splits; YOLO any-detection + TP vs label)")
    print("=" * 80)

    by_cam: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_cam[r["camera"]].append(r)

    for cam in sorted(by_cam.keys()):
        cr = by_cam[cam]
        n = len(cr)
        det = sum(int(r["detected"]) for r in cr)
        with_gt = [r for r in cr if int(r.get("has_gt_label", 0))]
        tp = sum(int(r["tp"]) for r in with_gt)
        print(f"\n{cam}: frames={n}  any_det={det} ({det/n*100:.1f}%)", end="")
        if with_gt:
            print(f"  TP@IoU={tp}/{len(with_gt)} ({tp/len(with_gt)*100:.1f}%)")
        else:
            print()

        banded = [r for r in cr if r.get("gt_distance_m") not in ("", None)]
        if not banded:
            continue
        print(f"  {'Band (m)':>12s}  {'n':>6s}  {'Det%':>7s}  {'TP%':>7s}")
        for lo, hi in BANDS:
            ib = [
                r
                for r in banded
                if lo <= float(r["gt_distance_m"]) < hi
            ]
            if not ib:
                continue
            d = sum(int(r["detected"]) for r in ib)
            t = sum(int(r["tp"]) for r in ib)
            print(
                f"  {lo:3d}-{hi:<3d}      {len(ib):>6d}  {d/len(ib)*100:6.1f}%  {t/len(ib)*100:6.1f}%"
            )

    print("=" * 80)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare sensors (RGB / LiDAR / Radar)")
    ap.add_argument("--runs-dir", type=str, default="runs/detect", help="Detection outputs folder")
    ap.add_argument(
        "--primary-rgb-camera",
        type=str,
        default="",
        help="Camera name matching detection_rgb_<name>.csv for aligned table",
    )
    ap.add_argument(
        "--legacy-rgb",
        action="store_true",
        help="Use detection_results.csv instead of detection_rgb_*.csv",
    )
    args = ap.parse_args()

    runs_dir = (ROOT / args.runs_dir).resolve() if not Path(args.runs_dir).is_absolute() else Path(args.runs_dir)

    rgb_path = pick_primary_rgb_path(runs_dir, args.primary_rgb_camera or None, args.legacy_rgb)
    rgb_label = "legacy_csv" if args.legacy_rgb else (rgb_path.stem.replace("detection_rgb_", "") if rgb_path else "none")

    table = build_frame_table(
        runs_dir,
        args.primary_rgb_camera or None,
        args.legacy_rgb,
    )

    summarise_multi_camera_rgb(runs_dir)

    if not table:
        print(
            "\nNo aligned frame table (need telemetry gt_distance_m in radar/lidar/rgb rows "
            "and overlapping keys). Run detect_lidar, detect_radar on sim dataset, or pass "
            "--sim-dataset when running detect_rgb_cameras."
        )
        return

    summarise_by_band(table, rgb_label=rgb_label)
    summarise_by_condition(table, rgb_label=rgb_label)
    summarise_first_detection(table)
    export_summary_csv(table, runs_dir, rgb_label)


if __name__ == "__main__":
    main()
