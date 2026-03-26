"""
generate_yolo_labels.py — Auto-generate YOLO bounding-box labels from AirSim
segmentation masks and prepare the dataset_yolo/ folder structure for training.

Usage:
    python tools/generate_yolo_labels.py --dataset dataset --output dataset_yolo --val-ratio 0.2

Alternativa sem segmentação (pose da câmera + posição do intruso em telemetry.csv):
    python tools/yolo_labels_from_projection.py --dataset dataset --output dataset_yolo_proj
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
import shutil
from pathlib import Path

import cv2
import numpy as np

# Cores BGR do intruso na seg (OpenCV). Referencia do casco: hull #5FDF9F = RGB (95,223,159) = BGR (159,223,95).
# O mesmo mesh pode aparecer noutras BGR (sub-meshes / shader); unimos todas para a bbox.
DEFAULT_DRONE_BGR_LIST: list[np.ndarray] = [
    np.array([159, 223, 95], dtype=np.uint8),  # hull #5FDF9F
    np.array([95, 223, 159], dtype=np.uint8),
    np.array([159, 223, 159], dtype=np.uint8),
    np.array([95, 127, 127], dtype=np.uint8),
    np.array([95, 127, 191], dtype=np.uint8),
    np.array([95, 127, 255], dtype=np.uint8),
    np.array([95, 191, 255], dtype=np.uint8),
    np.array([255, 223, 95], dtype=np.uint8),
    np.array([255, 223, 191], dtype=np.uint8),
    np.array([255, 223, 223], dtype=np.uint8),
    np.array([191, 223, 223], dtype=np.uint8),
    np.array([159, 255, 127], dtype=np.uint8),
]

MIN_PIXELS_DEFAULT = 20  # ignora manchas pequenas (ruido / cor igual noutro sitio)
MAX_BOX_AREA_FRAC_DEFAULT = 0.32  # rejeita caixa > frac da imagem (pixeis espurios longe do drone)


def build_drone_mask(seg_bgr: np.ndarray, drone_bgr_list: list[np.ndarray]) -> np.ndarray:
    """União de máscaras para cada cor BGR conhecida do drone."""
    mask = np.zeros(seg_bgr.shape[:2], dtype=bool)
    for c in drone_bgr_list:
        mask |= np.all(seg_bgr == c, axis=-1)
    return mask


def mask_largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """Mantem so a maior regiao 8-conexa (evita bbox gigante por pixeis BGR iguais espalhados)."""
    if mask.dtype != bool or not mask.any():
        return mask
    m = mask.astype(np.uint8)
    nlab, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if nlab <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + int(np.argmax(areas))
    return labels == best


def build_final_drone_bgr_list(
    extra_rows: list[list[int]] | None,
    only_custom: bool,
) -> list[np.ndarray]:
    """Junta lista por defeito com --drone-bgr, ou só custom com --only-custom-drone-bgr."""
    custom = [np.array([b, g, r], dtype=np.uint8) for b, g, r in (extra_rows or [])]
    if only_custom:
        return custom if custom else list(DEFAULT_DRONE_BGR_LIST)
    seen: set[tuple[int, int, int]] = set()
    out: list[np.ndarray] = []
    for arr in list(DEFAULT_DRONE_BGR_LIST) + custom:
        t = (int(arr[0]), int(arr[1]), int(arr[2]))
        if t not in seen:
            seen.add(t)
            out.append(arr)
    return out


def scan_seg_colors(dataset_dir: Path, max_images: int) -> None:
    """Imprime BGR (OpenCV) agregado nas primeiras máscaras seg — usar para escolher --drone-bgr."""
    runs = find_run_dirs(dataset_dir)
    if not runs:
        print(f"Sem run_* com seg/ em {dataset_dir}")
        return
    all_seg: list[Path] = []
    for run_dir in runs:
        all_seg.extend(sorted((run_dir / "seg").glob("frame_*.png")))
    paths = all_seg[: max(1, max_images)]
    if not paths:
        print("Sem PNGs em seg/")
        return

    ctr: Counter[tuple[int, int, int]] = Counter()
    for p in paths:
        im = cv2.imread(str(p))
        if im is None:
            continue
        flat = im.reshape(-1, 3)
        u, counts = np.unique(flat, axis=0, return_counts=True)
        for row, n in zip(u, counts):
            t = (int(row[0]), int(row[1]), int(row[2]))
            ctr[t] += int(n)

    print(f"(scan {len(paths)} imagens seg) - total {len(ctr)} cores unicas.\n")
    print("Mais frequentes (provável fundo / céu):")
    for t, n in ctr.most_common(8):
        print(f"  BGR {t}  pixels={n}")
    print("\nMenos frequentes (candidatos a objeto / drone) - copie B G R para --drone-bgr:")
    rare = sorted(ctr.items(), key=lambda x: x[1])[:35]
    for t, n in rare:
        b, g, r = t
        print(f"  --drone-bgr {b} {g} {r}   # {n} px")
    print(
        "\nNota: sem --only-custom-drone-bgr, cada --drone-bgr soma à lista por defeito. "
        "O hex do ecra (RGB) nao e BGR no ficheiro; use sempre estes valores do scan."
    )


def find_run_dirs(dataset_dir: Path) -> list[Path]:
    """run_* com rgb/ e seg/ (raiz do dataset ou subpastas tipo front_center/)."""
    out: list[Path] = []
    for p in sorted(dataset_dir.rglob("*")):
        if not p.is_dir() or not p.name.startswith("run_"):
            continue
        if (p / "rgb").is_dir() and (p / "seg").is_dir():
            out.append(p)
    return out


def extract_bbox(
    seg_path: Path,
    rgb_path: Path,
    drone_bgr_list: list[np.ndarray],
    *,
    min_pixels: int = MIN_PIXELS_DEFAULT,
    max_box_area_frac: float = MAX_BOX_AREA_FRAC_DEFAULT,
    largest_component: bool = True,
) -> tuple[float, float, float, float] | None:
    """YOLO (cx, cy, w, h) normalizado à resolução do RGB; escala seg → rgb pelos tamanhos reais."""
    seg = cv2.imread(str(seg_path))
    rgb = cv2.imread(str(rgb_path))
    if seg is None or rgb is None:
        return None

    rh, rw = rgb.shape[:2]
    sh, sw = seg.shape[:2]
    if sh < 1 or sw < 1:
        return None

    drone_mask = build_drone_mask(seg, drone_bgr_list)
    if largest_component:
        drone_mask = mask_largest_connected_component(drone_mask)
    npx = int(drone_mask.sum())
    if npx < min_pixels:
        return None

    ys, xs = np.where(drone_mask)
    sx = rw / float(sw)
    sy = rh / float(sh)
    x1 = float(xs.min()) * sx
    x2 = float(xs.max() + 1) * sx
    y1 = float(ys.min()) * sy
    y2 = float(ys.max() + 1) * sy

    cx = (x1 + x2) / 2.0 / rw
    cy = (y1 + y2) / 2.0 / rh
    w = (x2 - x1) / rw
    h = (y2 - y1) / rh
    if max_box_area_frac > 0 and (w * h) > max_box_area_frac:
        return None
    return cx, cy, w, h


def load_distances(telemetry_csv: Path) -> dict[int, float]:
    """Map frame index → distance_xy_m from telemetry."""
    distances: dict[int, float] = {}
    with open(telemetry_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            distances[int(row["frame"])] = float(row["dist_xy_m"])
    return distances


def process_dataset(
    dataset_dir: Path,
    output_dir: Path,
    val_ratio: float,
    seed: int,
    drone_bgr_list: list[np.ndarray],
    *,
    min_pixels: int = MIN_PIXELS_DEFAULT,
    max_box_area_frac: float = MAX_BOX_AREA_FRAC_DEFAULT,
    largest_component: bool = True,
):
    random.seed(seed)

    runs = find_run_dirs(dataset_dir)
    if not runs:
        print(f"No run_* folders with rgb/ and seg/ under {dataset_dir}")
        return

    # Collect all valid (image_path, label_str, distance) tuples.
    samples: list[tuple[Path, str, float]] = []
    skipped = 0
    warned_shape_runs: set[str] = set()

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

            rk = str(run_dir.resolve())
            if rk not in warned_shape_runs:
                _r = cv2.imread(str(rgb_file))
                _s = cv2.imread(str(seg_file))
                if (
                    _r is not None
                    and _s is not None
                    and _r.shape[:2] != _s.shape[:2]
                ):
                    warned_shape_runs.add(rk)
                    rh, rw = _r.shape[:2]
                    sh, sw = _s.shape[:2]
                    print(
                        f"[AVISO] {run_dir.name}: rgb {rw}x{rh} != seg {sw}x{sh}. "
                        "A caixa e escalada por tamanho; se o FOV da seg nao for o do RGB, "
                        "a caixa nao coincide com o RGB. Alinhe CaptureSettings ImageType 5 com ImageType 0 "
                        "no settings.json do Observer e regenere o dataset."
                    )

            bbox = extract_bbox(
                seg_file,
                rgb_file,
                drone_bgr_list,
                min_pixels=min_pixels,
                max_box_area_frac=max_box_area_frac,
                largest_component=largest_component,
            )
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
            # Unique id: rel path from dataset root (multicam: front_center/run_001_.../rgb -> prefix)
            run_dir = rgb_path.parent.parent
            try:
                rel = run_dir.resolve().relative_to(dataset_dir.resolve())
            except ValueError:
                rel = Path(run_dir.name)
            fname = "_".join(rel.parts) + "_" + rgb_path.stem
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
    p.add_argument(
        "--drone-bgr",
        nargs=3,
        type=int,
        metavar=("B", "G", "R"),
        action="append",
        dest="drone_bgr_rows",
        help="Somado à lista por defeito (OpenCV BGR). Repita para várias cores. "
             "Ver cores reais: --scan-seg-colors. Só estas cores: --only-custom-drone-bgr.",
    )
    p.add_argument(
        "--only-custom-drone-bgr",
        action="store_true",
        help="Usar apenas cores de --drone-bgr (ignora lista por defeito).",
    )
    p.add_argument(
        "--scan-seg-colors",
        type=int,
        nargs="?",
        const=50,
        default=None,
        metavar="N",
        help="Só diagnóstico: agrega BGR nas primeiras N imagens seg (omitir N => 50). Não grava YOLO.",
    )
    p.add_argument(
        "--min-drone-pixels",
        type=int,
        default=MIN_PIXELS_DEFAULT,
        metavar="N",
        help="Minimo de pixeis na mascara (apos maior componente) para aceitar frame.",
    )
    p.add_argument(
        "--max-box-area-frac",
        type=float,
        default=MAX_BOX_AREA_FRAC_DEFAULT,
        metavar="F",
        help="Rejeita label se (w*h) normalizado > F (caixas absurdas por ruído longe do drone). 0=desliga.",
    )
    p.add_argument(
        "--no-largest-seg-component",
        action="store_true",
        help="Usar todos os pixeis da cor (comportamento antigo; bbox pode expandir).",
    )
    args = p.parse_args()

    if args.scan_seg_colors is not None:
        scan_seg_colors(Path(args.dataset), max_images=max(1, args.scan_seg_colors))
        return

    colors = build_final_drone_bgr_list(args.drone_bgr_rows, args.only_custom_drone_bgr)
    print(f"Drone BGR cores usadas ({len(colors)}): {[tuple(c.tolist()) for c in colors]}")

    process_dataset(
        dataset_dir=Path(args.dataset),
        output_dir=Path(args.output),
        val_ratio=args.val_ratio,
        seed=args.seed,
        drone_bgr_list=colors,
        min_pixels=max(1, args.min_drone_pixels),
        max_box_area_frac=max(0.0, args.max_box_area_frac),
        largest_component=not args.no_largest_seg_component,
    )


if __name__ == "__main__":
    main()
