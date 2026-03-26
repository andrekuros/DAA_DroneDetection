"""
Experiment-30 — 30 cenarios (exp30_*) em scenarios.py: variacao de nivel, azimute, clima
(fog <= 0.1 quando usada), vento, hora, offset de yaw do Observer.

Captura: RGB + depth + seg + IR, Lidar VLP-16 + OS128, Radar (Echo), telemetria CSV por run.
Geometria alinhada ao Block: --rotate-observer, --vertical-separation (>=1 m), --no-avoid,
--approach-lateral-offset 0.

Uso:
  python tools/run_experiment30_dataset.py --capture
  python tools/run_experiment30_dataset.py --capture --multicam
  # So as outras cameras (sem repetir front_center):
  python tools/run_experiment30_dataset.py --capture --multicam --multicam-exclude front_center
  python tools/run_experiment30_dataset.py --only-post
  # So front_center apos captura multicam (YOLO + previews):
  python tools/run_experiment30_dataset.py --only-post-front-center

Requer settings.json do Observer com CaptureSettings alinhadas por ImageType (0,1,5,7) e sensores.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Import apos path (executar como python tools/run_experiment30_dataset.py)
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))

from run_block_dataset import (  # noqa: E402
    BLOCK_APPROACH_LATERAL_OFFSET_M,
    BLOCK_HULL_BGR,
    BLOCK_VERTICAL_SEPARATION_M,
    MULTICAM_DEFAULT,
    _copy_examples,
    _copy_examples_by_camera,
    _has_capture_data,
    _vertical_separation_mm,
)
from scenarios import EXPERIMENT30_SCENARIO_NAMES  # noqa: E402

DEFAULT_DATASET = Path("dataset_exp30")
DEFAULT_YOLO = Path("dataset_exp30_yolo")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Experiment-30: 30 cenarios, todos os sensores de imagem + Lidar + Radar"
    )
    ap.add_argument("--capture", action="store_true", help="Corre experiment_controller (Cosys-AirSim)")
    ap.add_argument(
        "--only-post",
        action="store_true",
        help="So generate_yolo_labels + draw_yolo + exemplos",
    )
    ap.add_argument(
        "--only-post-front-center",
        action="store_true",
        help="Igual a --only-post mas so dataset_exp30_multicam/front_center -> dataset_exp30_frontcenter_yolo",
    )
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Pasta de saida da captura")
    ap.add_argument("--yolo-out", type=Path, default=DEFAULT_YOLO, help="Dataset YOLO de saida")
    ap.add_argument("--max-frames", type=int, default=260, metavar="N")
    ap.add_argument("--stop-dist", type=float, default=22.0, metavar="M")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--examples", type=int, default=12)
    ap.add_argument("--ip", type=str, default="127.0.0.1")
    ap.add_argument("--multicam", action="store_true", help=f"Cameras: {list(MULTICAM_DEFAULT)}; pastas *_multicam")
    ap.add_argument(
        "--multicam-exclude",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Com --multicam e sem --cameras: tira estes nomes da lista por defeito (ex.: front_center).",
    )
    ap.add_argument("--cameras", nargs="+", default=None, metavar="NAME")
    ap.add_argument("--examples-per-camera", type=int, default=2, metavar="N")
    ap.add_argument(
        "--vertical-separation",
        type=_vertical_separation_mm,
        default=max(1.0, BLOCK_VERTICAL_SEPARATION_M + 0.2),
        metavar="M",
        help="Separacao NED Z+ do intruso (min 1.0 m); predefinido levemente > Block.",
    )
    ap.add_argument(
        "--full-mesh-bgr",
        action="store_true",
        help="YOLO com todas as cores do mesh (predefinido: so casco).",
    )
    ap.add_argument(
        "--no-lidar-radar",
        action="store_true",
        help="Nao passar --lidar-vlp16/--lidar-os128/--radar (so imagens).",
    )
    ap.add_argument(
        "--no-ir",
        action="store_true",
        help="Nao capturar IR (ImageType 7) se o settings.json nao tiver entrada IR na camera.",
    )
    args = ap.parse_args()

    if args.only_post_front_center:
        args.only_post = True
        args.dataset = Path("dataset_exp30_multicam/front_center")
        args.yolo_out = Path("dataset_exp30_frontcenter_yolo")

    if args.multicam:
        if args.cameras is None:
            cams = list(MULTICAM_DEFAULT)
            if args.multicam_exclude:
                ex = set(args.multicam_exclude)
                cams = [c for c in cams if c not in ex]
            if not cams:
                print("[ERRO] --multicam-exclude removeu todas as cameras.")
                sys.exit(2)
            args.cameras = cams
        if args.dataset == DEFAULT_DATASET:
            args.dataset = Path("dataset_exp30_multicam")
        if args.yolo_out == DEFAULT_YOLO:
            args.yolo_out = Path("dataset_exp30_multicam_yolo")

    dataset_abs = (REPO_ROOT / args.dataset).resolve()
    yolo_abs = (REPO_ROOT / args.yolo_out).resolve()
    py = sys.executable

    if args.capture:
        img_types = ["scene", "depth", "seg"]
        if not args.no_ir:
            img_types.append("ir")
        cmd = [
            py,
            str(REPO_ROOT / "tools" / "experiment_controller.py"),
            "--scenarios",
            *EXPERIMENT30_SCENARIO_NAMES,
            "--rotate-observer",
            "--output-dir",
            str(dataset_abs),
            "--max-frames",
            str(args.max_frames),
            "--stop-dist",
            str(args.stop_dist),
            "--image-types",
            *img_types,
            "--vertical-separation",
            str(args.vertical_separation),
            "--approach-lateral-offset",
            str(BLOCK_APPROACH_LATERAL_OFFSET_M),
            "--no-avoid",
            "--ip",
            args.ip,
        ]
        if not args.no_lidar_radar:
            cmd.extend(["--lidar-vlp16", "--lidar-os128", "--radar"])
        if args.cameras:
            cmd.extend(["--cameras", *args.cameras])
        print("Executando captura (30 cenarios x sensores completos):\n ", " ".join(cmd))
        r = subprocess.run(cmd, cwd=REPO_ROOT)
        if r.returncode != 0:
            print(f"[AVISO] experiment_controller saiu com codigo {r.returncode}")

    if not _has_capture_data(dataset_abs):
        print(f"[SKIP] Sem run_* com rgb/ e seg/ em {dataset_abs}. Use --capture.")
        return

    gen_yolo_cmd = [
        py,
        str(REPO_ROOT / "tools" / "generate_yolo_labels.py"),
        "--dataset",
        str(dataset_abs),
        "--output",
        str(yolo_abs),
        "--val-ratio",
        str(args.val_ratio),
        "--seed",
        str(args.seed),
    ]
    if not args.full_mesh_bgr:
        b, g, r = BLOCK_HULL_BGR
        gen_yolo_cmd.extend(["--only-custom-drone-bgr", "--drone-bgr", str(b), str(g), str(r)])
    subprocess.run(gen_yolo_cmd, cwd=REPO_ROOT, check=False)

    subprocess.run(
        [
            py,
            str(REPO_ROOT / "tools" / "draw_yolo_on_images.py"),
            "--yolo-root",
            str(yolo_abs),
            "--splits",
            "train",
            "val",
        ],
        cwd=REPO_ROOT,
        check=False,
    )

    _copy_examples(yolo_abs, args.examples)
    if args.cameras and args.examples_per_camera > 0:
        _copy_examples_by_camera(yolo_abs, tuple(args.cameras), args.examples_per_camera)


if __name__ == "__main__":
    main()
