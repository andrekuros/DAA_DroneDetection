"""
Compara caixas YOLO geradas por projeção 3D (telemetry) com caixas derivadas da
máscara de segmentação (quando o drone aparece na seg).

Também pode gravar sobreposições RGB para inspeção visual.

Captura (com Cosys-AirSim em execução), depois labels e verificação:

    python tools/experiment_controller.py \\
        --scenarios labeltest_clear_day labeltest_fog_dusk labeltest_rain_dawn \\
        --output-dir dataset_labeltest --max-frames 50

    python tools/yolo_labels_from_projection.py \\
        --dataset dataset_labeltest --output dataset_labeltest_yolo --val-ratio 0.0

    python tools/verify_projection_vs_seg.py --dataset dataset_labeltest --save-overlays 12
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import generate_yolo_labels as gyl  # noqa: E402
import yolo_labels_from_projection as ylp  # noqa: E402


def seg_bbox_yolo_normalized(seg_path: Path, rgb_w: int, rgb_h: int) -> tuple[float, float, float, float] | None:
    """BBox YOLO normalizado no espaço do RGB, escalando a seg para o tamanho do RGB."""
    img = cv2.imread(str(seg_path))
    if img is None:
        return None
    sh, sw = img.shape[:2]
    if sh < 1 or sw < 1:
        return None
    drone_mask = gyl.build_drone_mask(img, gyl.DEFAULT_DRONE_BGR_LIST)
    if drone_mask.sum() < 2:
        return None
    ys, xs = np.where(drone_mask)
    sx = rgb_w / float(sw)
    sy = rgb_h / float(sh)
    x1 = float(xs.min()) * sx
    x2 = float(xs.max() + 1) * sx
    y1 = float(ys.min()) * sy
    y2 = float(ys.max() + 1) * sy
    cx = (x1 + x2) / 2.0 / rgb_w
    cy = (y1 + y2) / 2.0 / rgb_h
    wn = (x2 - x1) / rgb_w
    hn = (y2 - y1) / rgb_h
    return cx, cy, wn, hn


def yolo_norm_to_xyxy(cx: float, cy: float, w: float, h: float, W: int, H: int) -> tuple[float, float, float, float]:
    px_w, px_h = w * W, h * H
    px_cx, px_cy = cx * W, cy * H
    x1 = px_cx - px_w / 2.0
    y1 = px_cy - px_h / 2.0
    x2 = px_cx + px_w / 2.0
    y2 = px_cy + px_h / 2.0
    return x1, y1, x2, y2


def iou_xyxy(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ab = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + ab - inter
    return inter / union if union > 0 else 0.0


def draw_yolo_box(
    im: np.ndarray, cx: float, cy: float, w: float, h: float, bgr: tuple[int, int, int], label: str
) -> None:
    H, W = im.shape[:2]
    x1, y1, x2, y2 = yolo_norm_to_xyxy(cx, cy, w, h, W, H)
    p1 = (int(round(x1)), int(round(y1)))
    p2 = (int(round(x2)), int(round(y2)))
    cv2.rectangle(im, p1, p2, bgr, 2)
    cv2.putText(im, label, (p1[0], max(20, p1[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 1, cv2.LINE_AA)


def main() -> None:
    p = argparse.ArgumentParser(description="Compara bbox projeção vs segmentação")
    p.add_argument("--dataset", type=Path, default=Path("dataset_labeltest"), help="Raiz com run_*/rgb, seg, telemetry.csv")
    p.add_argument("--fov-deg", type=float, default=None, help="Sobrescreve FOV (graus) como em yolo_labels_from_projection")
    p.add_argument("--half-span-m", type=float, default=0.35)
    p.add_argument("--default-fov-deg", type=float, default=120.0)
    p.add_argument("--save-overlays", type=int, default=0, metavar="N", help="Grava N imagens com as duas caixas em --overlay-dir")
    p.add_argument("--overlay-dir", type=Path, default=Path("label_verify_overlays"))
    p.add_argument(
        "--flip-image-v",
        action="store_true",
        help="Igual a yolo_labels_from_projection --flip-image-v.",
    )
    p.add_argument(
        "--v-nudge-pixels",
        type=float,
        default=0.0,
        metavar="PX",
        help="Igual a yolo_labels_from_projection --v-nudge-pixels.",
    )
    args = p.parse_args()

    half_ned = (args.half_span_m, args.half_span_m, args.half_span_m)
    rgb_files = sorted(args.dataset.rglob("rgb/frame_*.png"))
    ious: list[float] = []
    both = 0
    seg_only = 0
    proj_only = 0
    overlays_left = max(0, args.save_overlays)
    args.overlay_dir.mkdir(parents=True, exist_ok=True)

    for rgb_path in rgb_files:
        run_dir = rgb_path.parent.parent
        if not run_dir.name.startswith("run_"):
            continue
        seg_path = run_dir / "seg" / f"{rgb_path.stem}.png"
        tele_csv = run_dir / "telemetry.csv"
        if not tele_csv.exists():
            continue

        by_f = ylp.load_telemetry_by_frame(tele_csv)
        try:
            frame_idx = int(rgb_path.stem.replace("frame_", ""))
        except ValueError:
            continue
        row = by_f.get(frame_idx)
        if row is None:
            continue

        im = cv2.imread(str(rgb_path))
        if im is None:
            continue
        H, W = im.shape[:2]

        qw = ylp._parse_opt_float(row, "cam_qw")
        qx = ylp._parse_opt_float(row, "cam_qx")
        qy = ylp._parse_opt_float(row, "cam_qy")
        qz = ylp._parse_opt_float(row, "cam_qz")
        px = ylp._parse_opt_float(row, "cam_pos_x")
        py = ylp._parse_opt_float(row, "cam_pos_y")
        pz = ylp._parse_opt_float(row, "cam_pos_z")
        ix = ylp._parse_opt_float(row, "intr_global_x")
        iy = ylp._parse_opt_float(row, "intr_global_y")
        iz = ylp._parse_opt_float(row, "intr_global_z")

        proj_box = None
        if None not in (qw, qx, qy, qz, px, py, pz, ix, iy, iz):
            fov = args.fov_deg if args.fov_deg is not None else ylp._parse_opt_float(row, "cam_fov_deg")
            if fov is None or fov <= 0:
                fov = args.default_fov_deg
            proj_box = ylp.project_corners_to_yolo(
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
                flip_image_v=args.flip_image_v,
                v_nudge_px=args.v_nudge_pixels,
            )

        seg_box = None
        if seg_path.exists():
            seg_box = seg_bbox_yolo_normalized(seg_path, W, H)

        if proj_box and seg_box:
            both += 1
            pcx, pcy, pw, ph = proj_box
            scx, scy, sw, sh = seg_box
            iou = iou_xyxy(
                yolo_norm_to_xyxy(pcx, pcy, pw, ph, W, H),
                yolo_norm_to_xyxy(scx, scy, sw, sh, W, H),
            )
            ious.append(iou)
        elif proj_box:
            proj_only += 1
        elif seg_box:
            seg_only += 1

        if overlays_left > 0 and proj_box and seg_path.exists() and seg_box:
            vis = im.copy()
            pcx, pcy, pw, ph = proj_box
            scx, scy, sw, sh = seg_box
            draw_yolo_box(vis, pcx, pcy, pw, ph, (0, 255, 0), "proj")
            draw_yolo_box(vis, scx, scy, sw, sh, (0, 165, 255), "seg")
            out = args.overlay_dir / f"{run_dir.name}_{rgb_path.stem}.png"
            cv2.imwrite(str(out), vis)
            overlays_left -= 1

    n_rgb = len(rgb_files)
    print(f"RGB frames encontrados: {n_rgb}")
    print(f"Frames com projeção e seg válidos: {both}  |  só projeção: {proj_only}  |  só seg: {seg_only}")
    if ious:
        print(
            f"IoU projeção vs seg — min={min(ious):.3f}  max={max(ious):.3f}  "
            f"mean={float(np.mean(ious)):.3f}  median={float(np.median(ious)):.3f}"
        )
        ge5 = sum(1 for x in ious if x >= 0.5)
        print(f"IoU >= 0.5: {ge5}/{len(ious)} ({100.0 * ge5 / len(ious):.1f}%)")
    elif n_rgb == 0:
        print("Nenhum frame. Rode a captura com experiment_controller (ver docstring no topo).")
    else:
        print(
            "Sem pares comparáveis (falta telemetry com cam_*, ou seg sem drone BGR, ou só um dos dois). "
            "Confirme captura recente com experiment_controller atualizado e seg habilitada."
        )
    if args.save_overlays > 0:
        saved = args.save_overlays - overlays_left
        print(f"Sobreposições gravadas em: {args.overlay_dir.resolve()} ({saved} ficheiros)")


if __name__ == "__main__":
    main()
