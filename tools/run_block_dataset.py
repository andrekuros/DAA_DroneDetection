"""
Block Dataset — três condições (clima/hora) com o Observer a rodar (yaw) sobre a mesma geometria
de aproximação; labels YOLO a partir da segmentação; pré-visualizações com caixas.

1) Captura (simulador ligado):
    python tools/run_block_dataset.py --capture
    # Intruso pelo menos 1 m noutro nível (NED) que o plano do cenário (--vertical-separation);
    # voo em linha reta (--no-avoid, --approach-lateral-offset 0).

2) Várias câmeras (settings com Observer.front_center, front_medium, …) + seg + caixas:
    python tools/run_block_dataset.py --capture --multicam
    # Grava em dataset_block_multicam/<camera>/run_*_…/ ; YOLO em dataset_block_multicam_yolo/
    # Exemplos por câmera: block_examples_by_camera/<camera>/

3) Só pós-processamento (já existe dataset_block/.../run_*):
    python tools/run_block_dataset.py --only-post

Saídas:
    dataset_block/front_center/run_*_{scenario}/   (scene, seg, telemetry) — câmera única
    dataset_block_yolo/                             (images/, labels/, dataset.yaml; bbox = casco na seg)
    dataset_block_yolo/labeled_preview/             (caixa = hull #5FDF9F na seg)
    dataset_block_yolo/block_examples/             (primeiras N para relatório)
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

BLOCK_SCENARIOS = [
    "block_rot_clear_day",
    "block_rot_fog_dusk",
    "block_rot_rain_dawn",
]

DEFAULT_DATASET = Path("dataset_block")
DEFAULT_YOLO = Path("dataset_block_yolo")

# Câmeras definidas em config/cosys_airsim_settings_extra_cameras.json (Observer)
MULTICAM_DEFAULT = (
    "front_center",
    "front_medium",
    "front_narrow_hd",
    "front_lowcost_640",
)

# NED: Z+ = baixo; intruso fica +M m em Z face ao ponto do cenário (>= 1 m de separação vertical típica).
BLOCK_VERTICAL_SEPARATION_M = 2.5
BLOCK_APPROACH_LATERAL_OFFSET_M = 2.5

# Seg -> YOLO: bbox só com pixeis do casco #5FDF9F (OpenCV B G R). Ver generate_yolo_labels.py.
BLOCK_HULL_BGR = (159, 223, 95)


def _has_capture_data(dataset_dir: Path) -> bool:
    for p in dataset_dir.rglob("*"):
        if not p.is_dir() or not p.name.startswith("run_"):
            continue
        if (p / "rgb").is_dir() and (p / "seg").is_dir():
            return True
    return False


def _vertical_separation_mm(s: str) -> float:
    v = float(s)
    if v < 1.0:
        raise argparse.ArgumentTypeError("must be >= 1.0 m (block: intruso noutro nivel que o plano do cenario)")
    return v


def _copy_examples(yolo_root: Path, n: int) -> None:
    if n <= 0:
        return
    ex_dir = yolo_root / "block_examples"
    ex_dir.mkdir(parents=True, exist_ok=True)
    for old in ex_dir.glob("*.png"):
        old.unlink()
    taken = 0
    for split in ("train", "val"):
        prev = yolo_root / "labeled_preview" / split
        if not prev.is_dir():
            continue
        for f in sorted(prev.glob("*.png")):
            if taken >= n:
                return
            shutil.copy2(f, ex_dir / f.name)
            taken += 1
    if taken:
        print(f"Exemplos (Block): {ex_dir} ({taken} ficheiros)")


def _copy_examples_by_camera(yolo_root: Path, cameras: tuple[str, ...], n_each: int) -> None:
    """Copia até n_each previews por câmera (prefixo fname: '<camera>_')."""
    if n_each <= 0 or not cameras:
        return
    ex_root = yolo_root / "block_examples_by_camera"
    if ex_root.exists():
        shutil.rmtree(ex_root)
    ex_root.mkdir(parents=True, exist_ok=True)
    for cam in cameras:
        dest = ex_root / cam
        dest.mkdir(parents=True, exist_ok=True)
        taken = 0
        for split in ("train", "val"):
            prev = yolo_root / "labeled_preview" / split
            if not prev.is_dir():
                continue
            for f in sorted(prev.glob("*.png")):
                if taken >= n_each:
                    break
                if f.name.startswith(f"{cam}_"):
                    shutil.copy2(f, dest / f.name)
                    taken += 1
            if taken >= n_each:
                break
        if taken:
            print(f"Exemplos [{cam}]: {dest} ({taken} ficheiros)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Block Dataset: rotate observer + seg -> YOLO + previews")
    ap.add_argument("--capture", action="store_true", help="Corre experiment_controller (Cosys-AirSim)")
    ap.add_argument(
        "--only-post",
        action="store_true",
        help="Só generate_yolo_labels + draw_yolo + exemplos (ignora impressão de comando de captura)",
    )
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Pasta de saída da captura")
    ap.add_argument("--yolo-out", type=Path, default=DEFAULT_YOLO, help="Dataset YOLO de saída")
    ap.add_argument(
        "--max-frames",
        type=int,
        default=240,
        metavar="N",
        help="~24s a 10Hz; 80m a 5m/s ~16s ate ao obs + margem para cruzar e passar atras",
    )
    ap.add_argument(
        "--stop-dist",
        type=float,
        default=20.0,
        metavar="M",
        help="Grava até o intruso estar ~M m além do plano do observador (eixo de aproximação).",
    )
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fixed-intruder-azimuth", type=float, default=0.0, metavar="DEG")
    ap.add_argument("--examples", type=int, default=9, help="N PNGs copiados para block_examples/")
    ap.add_argument("--ip", type=str, default="127.0.0.1")
    ap.add_argument(
        "--multicam",
        action="store_true",
        help=f"Usa câmeras {list(MULTICAM_DEFAULT)}; dataset/yolo por defeito *_multicam (alinhado a extra_cameras.json).",
    )
    ap.add_argument(
        "--cameras",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Lista de câmeras para experiment_controller (--capture). Ex.: front_center front_medium",
    )
    ap.add_argument(
        "--examples-per-camera",
        type=int,
        default=3,
        metavar="N",
        help="Com varias cameras, copia N previews para block_examples_by_camera/<camera>/ (0=desliga).",
    )
    ap.add_argument(
        "--vertical-separation",
        type=_vertical_separation_mm,
        default=BLOCK_VERTICAL_SEPARATION_M,
        metavar="M",
        help="Separacao vertical do intruso (m, NED Z+); minimo 1.0 m.",
    )
    ap.add_argument(
        "--full-mesh-bgr",
        action="store_true",
        help="generate_yolo_labels: unir todas as cores do mesh (predefinido: so casco BGR 159 223 95).",
    )
    args = ap.parse_args()

    if args.multicam:
        if args.cameras is None:
            args.cameras = list(MULTICAM_DEFAULT)
        if args.dataset == DEFAULT_DATASET:
            args.dataset = Path("dataset_block_multicam")
        if args.yolo_out == DEFAULT_YOLO:
            args.yolo_out = Path("dataset_block_multicam_yolo")

    dataset_abs = (REPO_ROOT / args.dataset).resolve()
    yolo_abs = (REPO_ROOT / args.yolo_out).resolve()
    py = sys.executable

    if args.capture:
        cmd = [
            py,
            str(REPO_ROOT / "tools" / "experiment_controller.py"),
            "--scenarios",
            *BLOCK_SCENARIOS,
            "--rotate-observer",
            "--output-dir",
            str(dataset_abs),
            "--max-frames",
            str(args.max_frames),
            "--stop-dist",
            str(args.stop_dist),
            "--image-types",
            "scene",
            "seg",
            "--vertical-separation",
            str(args.vertical_separation),
            "--approach-lateral-offset",
            str(BLOCK_APPROACH_LATERAL_OFFSET_M),
            "--no-avoid",
            "--ip",
            args.ip,
        ]
        if args.cameras:
            cmd.extend(["--cameras", *args.cameras])
        print("Executando captura:\n ", " ".join(cmd))
        r = subprocess.run(cmd, cwd=REPO_ROOT)
        if r.returncode != 0:
            print(f"[AVISO] experiment_controller saiu com código {r.returncode}")
    elif not args.only_post:
        print("Comando sugerido (simulador ligado):")
        cam_line = ""
        if args.cameras:
            cam_line = f"    --cameras {' '.join(args.cameras)} \\\n"
        print(
            f"  python tools/experiment_controller.py \\\n"
            f"    --scenarios {' '.join(BLOCK_SCENARIOS)} \\\n"
            f"    --rotate-observer \\\n"
            f"    --output-dir {args.dataset} --max-frames {args.max_frames} \\\n"
            f"    --stop-dist {args.stop_dist} \\\n"
            f"    --image-types scene seg \\\n"
            f"    --vertical-separation {args.vertical_separation} \\\n"
            f"    --approach-lateral-offset {BLOCK_APPROACH_LATERAL_OFFSET_M} \\\n"
            f"    --no-avoid \\\n"
            f"{cam_line}"
            f"    # multicam: python tools/run_block_dataset.py --capture --multicam"
        )

    if not _has_capture_data(dataset_abs):
        print(f"[SKIP] Sem run_* com rgb/ e seg/ em {dataset_abs}. Use --capture ou ajuste --dataset.")
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
        gen_yolo_cmd.extend(
            [
                "--only-custom-drone-bgr",
                "--drone-bgr",
                str(b),
                str(g),
                str(r),
            ]
        )
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
