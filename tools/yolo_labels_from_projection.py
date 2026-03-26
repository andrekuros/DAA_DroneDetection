"""
yolo_labels_from_projection.py — Gera labels YOLO (cx,cy,w,h) a partir da telemetria
AirSim/Cosys: pose da câmera + posição global do intruso, sem depender de segmentação.

Requer telemetry.csv com colunas gravadas pelo experiment_controller (img_*, cam_pos_*, cam_q*,
cam_fov_deg) e pastas rgb/ por run.

Uso:
    python tools/yolo_labels_from_projection.py --dataset dataset --output dataset_yolo_proj --val-ratio 0.2
    python tools/yolo_labels_from_projection.py --dataset dataset --fov-deg 120   # sobrescreve FOV do CSV
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


def _parse_opt_float(row: dict, key: str) -> float | None:
    v = row.get(key, "")
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def quat_body_to_world_R(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """R body→world NED: v_world = R @ v_body (eixo +X da câmera = frente)."""
    n = qw * qw + qx * qx + qy * qy + qz * qz
    if n < 1e-20:
        return np.eye(3)
    s = 2.0 / n
    wx, wy, wz = s * qw * qx, s * qw * qy, s * qw * qz
    xx, xy, xz = s * qx * qx, s * qx * qy, s * qx * qz
    yy, yz, zz = s * qy * qy, s * qy * qz, s * qz * qz
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float64,
    )


def intrinsics_horizontal_fov(W: int, H: int, fov_h_deg: float) -> tuple[float, float, float, float]:
    """FOV horizontal (graus), imagem W×H; pinhole AirSim: u = cx + fx·(y/x), v = cy + fy·(z/x) (+ nudge opcional)."""
    fh = math.radians(fov_h_deg)
    tan_h = math.tan(fh / 2.0)
    fx = (W / 2.0) / tan_h
    fov_v = 2.0 * math.atan(tan_h * H / W)
    fy = (H / 2.0) / math.tan(fov_v / 2.0)
    return W / 2.0, H / 2.0, fx, fy


def project_corners_to_yolo(
    cam_pos: tuple[float, float, float],
    qw: float,
    qx: float,
    qy: float,
    qz: float,
    intr_ned: tuple[float, float, float],
    half_ned: tuple[float, float, float],
    W: int,
    H: int,
    fov_h_deg: float,
    min_depth_m: float = 0.05,
    min_side_px: float = 2.0,
    *,
    flip_image_v: bool = False,
    v_nudge_px: float = 0.0,
) -> tuple[float, float, float, float] | None:
    """Por defeito v = cy + fy·(z/x) (frame câmera AirSim típico). --flip-image-v usa o sinal oposto (casos raros).
    v_nudge_px: desloca todas as projeções em +v (pixéis), p.ex. +4..+12 se a caixa ficar ligeiramente acima do drone."""
    R_bw = quat_body_to_world_R(qw, qx, qy, qz)
    R_wb = R_bw.T
    hx, hy, hz = half_ned
    intr = np.array(intr_ned, dtype=np.float64)
    cam = np.array(cam_pos, dtype=np.float64)
    us: list[float] = []
    vs: list[float] = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                p = intr + np.array([sx * hx, sy * hy, sz * hz], dtype=np.float64)
                d_w = p - cam
                d_c = R_wb @ d_w
                if d_c[0] < min_depth_m:
                    continue
                cx0, cy0, fx, fy = intrinsics_horizontal_fov(W, H, fov_h_deg)
                u = cx0 + fx * (d_c[1] / d_c[0])
                rz = d_c[2] / d_c[0]
                v = (cy0 - fy * rz) if flip_image_v else (cy0 + fy * rz)
                v += v_nudge_px
                us.append(u)
                vs.append(v)
    if len(us) < 2:
        return None
    u1, u2 = min(us), max(us)
    v1, v2 = min(vs), max(vs)
    u1 = float(np.clip(u1, 0.0, float(W)))
    u2 = float(np.clip(u2, 0.0, float(W)))
    v1 = float(np.clip(v1, 0.0, float(H)))
    v2 = float(np.clip(v2, 0.0, float(H)))
    if u2 - u1 < min_side_px or v2 - v1 < min_side_px:
        return None
    cx = (u1 + u2) / 2.0 / W
    cy = (v1 + v2) / 2.0 / H
    wn = (u2 - u1) / W
    hn = (v2 - v1) / H
    if wn <= 0 or hn <= 0:
        return None
    return cx, cy, wn, hn


def load_telemetry_by_frame(tele_csv: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    with open(tele_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[int(row["frame"])] = row
    return out


def process_dataset(
    dataset_dir: Path,
    output_dir: Path,
    val_ratio: float,
    seed: int,
    fov_override_deg: float | None,
    half_ned: tuple[float, float, float],
    default_fov_deg: float,
    flip_image_v: bool = False,
    v_nudge_px: float = 0.0,
):
    random.seed(seed)
    rgb_files = sorted(dataset_dir.rglob("rgb/frame_*.png"))
    samples: list[tuple[Path, str, float]] = []
    skipped_no_tele = 0
    skipped_pose = 0
    skipped_proj = 0

    for rgb_path in rgb_files:
        run_dir = rgb_path.parent.parent
        if not run_dir.name.startswith("run_"):
            continue
        tele_csv = run_dir / "telemetry.csv"
        if not tele_csv.exists():
            skipped_no_tele += 1
            continue
        by_f = load_telemetry_by_frame(tele_csv)
        stem = rgb_path.stem
        try:
            frame_idx = int(stem.replace("frame_", ""))
        except ValueError:
            continue
        row = by_f.get(frame_idx)
        if row is None:
            skipped_no_tele += 1
            continue

        im = cv2.imread(str(rgb_path))
        if im is None:
            continue
        H, W = im.shape[:2]
        w_csv = _parse_opt_float(row, "img_w")
        h_csv = _parse_opt_float(row, "img_h")
        if w_csv is not None and h_csv is not None and (int(w_csv) != W or int(h_csv) != H):
            # Telemetria de outra resolução — ainda projetamos no tamanho real da imagem.
            pass

        qw = _parse_opt_float(row, "cam_qw")
        qx = _parse_opt_float(row, "cam_qx")
        qy = _parse_opt_float(row, "cam_qy")
        qz = _parse_opt_float(row, "cam_qz")
        px = _parse_opt_float(row, "cam_pos_x")
        py = _parse_opt_float(row, "cam_pos_y")
        pz = _parse_opt_float(row, "cam_pos_z")
        ix = _parse_opt_float(row, "intr_global_x")
        iy = _parse_opt_float(row, "intr_global_y")
        iz = _parse_opt_float(row, "intr_global_z")
        if None in (qw, qx, qy, qz, px, py, pz, ix, iy, iz):
            skipped_pose += 1
            continue

        fov = fov_override_deg if fov_override_deg is not None else _parse_opt_float(row, "cam_fov_deg")
        if fov is None or fov <= 0:
            fov = default_fov_deg

        bbox = project_corners_to_yolo(
            (px, py, pz),
            qw,
            qx,
            qy,
            qz,
            (ix, iy, iz),
            half_ned,
            W,
            H,
            fov,
            flip_image_v=flip_image_v,
            v_nudge_px=v_nudge_px,
        )
        if bbox is None:
            skipped_proj += 1
            continue

        cx, cy, wn, hn = bbox
        label_line = f"0 {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}"
        dist = _parse_opt_float(row, "dist_xy_m") or -1.0
        samples.append((rgb_path, label_line, dist))

    print(
        f"Valid samples: {len(samples)}  |  "
        f"skipped: no_telemetry={skipped_no_tele} missing_pose={skipped_pose} "
        f"no_projection={skipped_proj}"
    )
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

        for rgb_path, label_line, _dist in split_samples:
            run_name = rgb_path.parent.parent.name
            fname = f"{run_name}_{rgb_path.stem}"
            shutil.copy2(rgb_path, img_dir / f"{fname}.png")
            (lbl_dir / f"{fname}.txt").write_text(label_line + "\n", encoding="utf-8")

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

    all_dists = [d for _, _, d in samples if d > 0]
    if all_dists:
        print(
            f"Distance range: {min(all_dists):.1f} – {max(all_dists):.1f} m  "
            f"(mean {np.mean(all_dists):.1f} m)"
        )


def main():
    p = argparse.ArgumentParser(
        description="YOLO labels from camera pose + intruder NED position (no segmentation)"
    )
    p.add_argument("--dataset", type=str, default="dataset", help="Raiz do dataset (runs com rgb/ e telemetry.csv)")
    p.add_argument("--output", type=str, default="dataset_yolo_proj", help="Saída estilo YOLO")
    p.add_argument("--val-ratio", type=float, default=0.2, help="Fração para validação")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--fov-deg",
        type=float,
        default=None,
        help="Força FOV horizontal em graus (ignora cam_fov_deg do CSV; útil se a API retornar 90 errado).",
    )
    p.add_argument(
        "--default-fov-deg",
        type=float,
        default=120.0,
        help="FOV usado se a linha do CSV não tiver cam_fov_deg.",
    )
    p.add_argument(
        "--half-span-m",
        type=float,
        default=0.35,
        help="Meia-extensão (m) da caixa NED alinhada aos eixos, igual em N/E/D (cubo aproximado do drone).",
    )
    p.add_argument(
        "--half-span-north",
        type=float,
        default=None,
        help="Sobrescreve só eixo North (m); requer também east/down se usar.",
    )
    p.add_argument("--half-span-east", type=float, default=None)
    p.add_argument("--half-span-down", type=float, default=None)
    p.add_argument(
        "--flip-image-v",
        action="store_true",
        help="Usa v = cy - fy·(z/x) em vez do default (+). Só se souber que o teu fork precisa disto.",
    )
    p.add_argument(
        "--v-nudge-pixels",
        type=float,
        default=0.0,
        metavar="PX",
        help="Soma PX à coordenada v em pixéis (positivo = caixa desce). Útil se a projeção ficar um pouco acima do drone.",
    )
    args = p.parse_args()

    h = args.half_span_m
    half_ned = (
        args.half_span_north if args.half_span_north is not None else h,
        args.half_span_east if args.half_span_east is not None else h,
        args.half_span_down if args.half_span_down is not None else h,
    )

    process_dataset(
        dataset_dir=Path(args.dataset),
        output_dir=Path(args.output),
        val_ratio=args.val_ratio,
        seed=args.seed,
        fov_override_deg=args.fov_deg,
        half_ned=half_ned,
        default_fov_deg=args.default_fov_deg,
        flip_image_v=args.flip_image_v,
        v_nudge_px=args.v_nudge_pixels,
    )


if __name__ == "__main__":
    main()
