"""
Build publication-oriented figures and a metrics JSON for multisensor_section.tex.

Reads Ultralytics results.csv under runs/detect/runs/train/<name>/.
Optionally runs YOLO val with the multicam best.pt on each camera YAML (needs CUDA for speed).

Usage:
  python tools/build_multisensor_report_figures.py
  python tools/build_multisensor_report_figures.py --no-cross-val
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS_TRAIN = ROOT / "runs" / "detect" / "runs" / "train"
FIGURES = ROOT / "figures"

RUNS: dict[str, dict] = {
    "multicam_pooled": {
        "dir": "exp30_multicam_yolov8s",
        "label": "Multicam (4 câmeras)",
        "imgsz": 960,
    },
    "narrow_hd": {
        "dir": "exp30_narrow_hd",
        "label": "Narrow HD",
        "imgsz": 960,
    },
    "medium": {
        "dir": "exp30_medium",
        "label": "Medium",
        "imgsz": 960,
    },
    "lowcost640": {
        "dir": "exp30_lowcost640",
        "label": "Lowcost 640",
        "imgsz": 640,
    },
}

DATASETS_FOR_CROSSVAL = [
    ("dataset_exp30_multicam_yolo/dataset.yaml", 960),
    ("dataset_exp30_yolo_narrow_hd/dataset.yaml", 960),
    ("dataset_exp30_yolo_medium/dataset.yaml", 960),
    ("dataset_exp30_yolo_lowcost640/dataset.yaml", 640),
]


def read_last_metrics(run_dir: Path) -> dict:
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    last = rows[-1]
    return {
        "epoch": int(float(last["epoch"])),
        "precision": float(last["metrics/precision(B)"]),
        "recall": float(last["metrics/recall(B)"]),
        "mAP50": float(last["metrics/mAP50(B)"]),
        "mAP50_95": float(last["metrics/mAP50-95(B)"]),
    }


def load_map_curve(run_dir: Path) -> tuple[list[int], list[float], list[float]]:
    csv_path = run_dir / "results.csv"
    epochs: list[int] = []
    m50: list[float] = []
    m5095: list[float] = []
    if not csv_path.exists():
        return epochs, m50, m5095
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            epochs.append(int(float(row["epoch"])))
            m50.append(float(row["metrics/mAP50(B)"]))
            m5095.append(float(row["metrics/mAP50-95(B)"]))
    return epochs, m50, m5095


def count_yolo_split(yaml_rel: str) -> tuple[int, int]:
    ydir = ROOT / Path(yaml_rel).parent
    train = ydir / "images" / "train"
    val = ydir / "images" / "val"
    nt = len(list(train.glob("*"))) if train.exists() else 0
    nv = len(list(val.glob("*"))) if val.exists() else 0
    return nt, nv


def cross_validate_multicam(device: str | int) -> list[dict]:
    from ultralytics import YOLO

    best = RUNS_TRAIN / "exp30_multicam_yolov8s" / "weights" / "best.pt"
    if not best.exists():
        return []
    model = YOLO(str(best))
    out: list[dict] = []
    for yaml_rel, imgsz in DATASETS_FOR_CROSSVAL:
        ypath = ROOT / yaml_rel
        if not ypath.exists():
            continue
        r = model.val(
            data=str(ypath),
            imgsz=imgsz,
            device=device,
            verbose=False,
            plots=False,
        )
        mp = float(r.box.map50)
        m95 = float(r.box.map)
        out.append(
            {
                "dataset": yaml_rel,
                "imgsz": imgsz,
                "precision": float(r.box.p.mean()),
                "recall": float(r.box.r.mean()),
                "mAP50": mp,
                "mAP50_95": m95,
            }
        )
    return out


def plot_bar_comparison(rows: list[dict], labels: list[str], outfile: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(len(labels))
    w = 0.2
    fig, ax = plt.subplots(figsize=(9.5, 5), dpi=150)
    ax.bar(x - 1.5 * w, [r["precision"] for r in rows], w, label="Precision")
    ax.bar(x - 0.5 * w, [r["recall"] for r in rows], w, label="Recall")
    ax.bar(x + 0.5 * w, [r["mAP50"] for r in rows], w, label="mAP50")
    ax.bar(x + 1.5 * w, [r["mAP50_95"] for r in rows], w, label="mAP50-95")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    ax.set_title("YOLOv8s — Experiment 30: validation metrics por treino (cosys-AirSim)")
    fig.tight_layout()
    fig.savefig(outfile.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(outfile.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def plot_map_curves(curves: list[tuple[str, Path]], outfile: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    for label, run_dir in curves:
        ep, _, m95 = load_map_curve(run_dir)
        if ep:
            ax.plot(ep, m95, label=label, linewidth=1.8)
    ax.set_xlabel("Época")
    ax.set_ylabel("mAP50-95 (validação)")
    ax.set_title("Convergência mAP50-95 — campanhas por câmera vs. multicam")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(outfile.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(outfile.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def plot_dataset_sizes(labels: list[str], trains: list[int], vals: list[int], outfile: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=150)
    ax.bar(x - 0.2, trains, 0.4, label="Train (caixas GT)")
    ax.bar(x + 0.2, vals, 0.4, label="Val")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Imagens com label")
    ax.set_title("Tamanho dos conjuntos YOLO (segmentação → bbox, hull BGR)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outfile.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(outfile.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def plot_sensor_distance_bars(outfile: Path) -> None:
    """Static recreation of report Table 2.3.4 style (Point D aligned methodology)."""
    import matplotlib.pyplot as plt
    import numpy as np

    bands = ["0–10", "10–25", "25–50", "50–100", "100–150", "150–250"]
    rgb = np.array([7.5, 2.8, 10.2, 6.7, 1.2, 0.0])
    os128 = np.array([77.6, 77.1, 67.7, 55.4, 58.5, 6.7])
    vlp16 = np.array([29.9, 32.1, 16.5, 11.2, 0.0, 0.0])
    radar = np.array([100.0] * 6)
    x = np.arange(len(bands))
    w = 0.2
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    ax.bar(x - 1.5 * w, rgb, w, label="RGB YOLO (ref. HD)")
    ax.bar(x - 0.5 * w, os128, w, label="LiDAR OS128")
    ax.bar(x + 0.5 * w, vlp16, w, label="LiDAR VLP-16")
    ax.bar(x + 1.5 * w, radar, w, label="Radar (presença)")
    ax.set_xticks(x)
    ax.set_xticklabels(bands)
    ax.set_ylabel("Taxa de deteção (%)")
    ax.set_xlabel("Banda de distância (m)")
    ax.set_title("Benchmark alinhado 640 frames — taxa por modalidade e distância")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(outfile.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(outfile.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cross-val", action="store_true")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    FIGURES.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    labels: list[str] = []
    curves: list[tuple[str, Path]] = []
    sizes_train: list[int] = []
    sizes_val: list[int] = []
    size_labels: list[str] = []

    for key, cfg in RUNS.items():
        rd = RUNS_TRAIN / cfg["dir"]
        m = read_last_metrics(rd)
        if not m:
            continue
        rows.append(m)
        labels.append(cfg["label"])
        curves.append((cfg["label"], rd))

    yaml_by_key = {
        "multicam_pooled": "dataset_exp30_multicam_yolo/dataset.yaml",
        "narrow_hd": "dataset_exp30_yolo_narrow_hd/dataset.yaml",
        "medium": "dataset_exp30_yolo_medium/dataset.yaml",
        "lowcost640": "dataset_exp30_yolo_lowcost640/dataset.yaml",
    }
    for key, cfg in RUNS.items():
        if key not in yaml_by_key:
            continue
        nt, nv = count_yolo_split(yaml_by_key[key])
        size_labels.append(cfg["label"])
        sizes_train.append(nt)
        sizes_val.append(nv)

    plot_bar_comparison(rows, labels, FIGURES / "exp30_yolo_camera_metrics")
    plot_map_curves(curves, FIGURES / "exp30_yolo_map_curves")
    plot_dataset_sizes(size_labels, sizes_train, sizes_val, FIGURES / "exp30_yolo_dataset_sizes")
    plot_sensor_distance_bars(FIGURES / "exp30_sensor_detection_by_distance")

    cross: list[dict] = []
    if not args.no_cross_val:
        try:
            cross = cross_validate_multicam(args.device)
        except Exception as e:
            print("[AVISO] cross-val multicam:", e)
            cross = []

    payload = {
        "runs": {lab: rows[i] for i, lab in enumerate(labels)},
        "dataset_splits": {
            size_labels[i]: {"train": sizes_train[i], "val": sizes_val[i]}
            for i in range(len(size_labels))
        },
        "crossval_multicam_best": cross,
    }
    out_json = FIGURES / "exp30_report_metrics.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("Wrote", out_json)
    print("Figures in", FIGURES.resolve())


if __name__ == "__main__":
    main()
