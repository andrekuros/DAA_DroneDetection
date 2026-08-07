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


def _sensor_band_arrays():
    import numpy as np

    bands = ["0–10", "10–25", "25–50", "50–100", "100–150", "150–250"]
    eo_unified = np.array([55.6, 44.3, 91.1, 100.0, 2.0, 0.0])
    os128 = np.array([77.6, 77.1, 67.7, 55.4, 58.5, 6.7])
    vlp16 = np.array([29.9, 32.1, 16.5, 11.2, 0.0, 0.0])
    radar = np.array([100.0] * 6)
    return bands, eo_unified, os128, vlp16, radar


def plot_sensor_distance_bars(outfile: Path) -> None:
    """Cross-sensor detection rate by distance band.
    EO rates: unified multi-camera YOLOv8s (any-camera-detects policy).
    LiDAR/Radar: from original synchronized benchmark (unchanged)."""
    import matplotlib.pyplot as plt
    import numpy as np

    bands, eo_unified, os128, vlp16, radar = _sensor_band_arrays()
    x = np.arange(len(bands))
    w = 0.2
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    ax.bar(x - 1.5 * w, eo_unified, w, label="EO/Vision (YOLOv8s, multi-cam)", color="#2ECC71")
    ax.bar(x - 0.5 * w, os128, w, label="LiDAR HD (OS128)", color="#3498DB")
    ax.bar(x + 0.5 * w, vlp16, w, label="LiDAR Sparse (VLP-16)", color="#9B59B6")
    ax.bar(x + 1.5 * w, radar, w, label="Radar (return-presence)", color="#E74C3C")
    ax.set_xticks(x)
    ax.set_xticklabels(bands)
    ax.set_ylabel("Detection rate (%)")
    ax.set_xlabel("Intruder distance band (m)")
    ax.set_title("Cross-Sensor Detection Rate by Distance — Exp. 30 Synchronized Benchmark")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, 108)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outfile.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(outfile.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def plot_wsc26_composite(outfile: Path) -> None:
    """WSC26 Fig.1: detection rate vs range (compact single panel for 2-page abstract)."""
    import matplotlib.pyplot as plt
    import numpy as np

    bands, eo_unified, os128, vlp16, radar = _sensor_band_arrays()
    x = np.arange(len(bands))
    w = 0.2

    fig, ax = plt.subplots(figsize=(6.0, 2.35), dpi=200)
    ax.bar(x - 1.5 * w, eo_unified, w, label="EO (YOLOv8s)", color="#2ECC71")
    ax.bar(x - 0.5 * w, os128, w, label="LiDAR HD", color="#3498DB")
    ax.bar(x + 0.5 * w, vlp16, w, label="LiDAR Sparse", color="#9B59B6")
    ax.bar(x + 1.5 * w, radar, w, label="Radar (presence)", color="#E74C3C")
    ax.set_xticks(x)
    ax.set_xticklabels(bands, fontsize=8)
    ax.set_ylabel("Detection rate (%)", fontsize=9)
    ax.set_xlabel("Intruder distance band (m)", fontsize=9)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9, ncol=2)
    fig.tight_layout()
    fig.savefig(outfile.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(outfile.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print("Wrote", outfile.with_suffix(".pdf"))


def _trim_letterbox(img):
    """Crop uniform black letterbox padding (e.g. 16:9 padded into 4:3)."""
    import numpy as np

    # Non-black mask: any channel above a low threshold.
    mask = img.max(axis=2) > 8
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return img
    return img[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _resize_h(img, th: int):
    """Resize image to target height, preserving aspect ratio."""
    from PIL import Image as _Image
    import numpy as np

    pil = _Image.fromarray(img)
    nw = int(round(pil.width * (th / pil.height)))
    return np.asarray(pil.resize((nw, th), _Image.Resampling.LANCZOS))


def _label_panel(img, text: str):
    """Draw a small opaque label in the top-left corner."""
    import cv2
    import numpy as np

    out = img.copy()
    if out.dtype != np.uint8:
        out = out.astype(np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.45, min(out.shape[:2]) / 900)
    thick = max(1, int(round(scale * 2)))
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    pad = 4
    cv2.rectangle(out, (4, 4), (4 + tw + 2 * pad, 4 + th + 2 * pad), (0, 0, 0), -1)
    cv2.putText(out, text, (4 + pad, 4 + th + pad - 1), font, scale, (0, 220, 255), thick, cv2.LINE_AA)
    return out


def plot_wsc26_urban_panels(outfile: Path, source: Path | None = None) -> None:
    """WSC26 Fig.2: horizontal 1x4 — cameras (left pair) then weather (right pair)."""
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    cam_src = source or (FIGURES / "shot.png")
    wx_src = FIGURES / "exp30_encounter_conditions.png"
    if not cam_src.exists():
        raise FileNotFoundError(f"Urban composite not found: {cam_src}")
    if not wx_src.exists():
        raise FileNotFoundError(f"Weather grid not found: {wx_src}")

    cam = np.asarray(Image.open(cam_src).convert("RGB"))
    ch, cw = cam.shape[:2]
    mid_y, mid_x = ch // 2, cw // 2
    # shot.png 2x2: TL Standard, TR Narrow HD, BL Medium, BR Low-Cost.
    narrow = _trim_letterbox(cam[:mid_y, mid_x:])
    lowcost = _trim_letterbox(cam[mid_y:, mid_x:])

    wx = np.asarray(Image.open(wx_src).convert("RGB"))
    wh, ww = wx.shape[:2]
    # exp30_encounter_conditions.png is 2x3:
    # TL Clear day, TM Dusk, TR Rain / BL Night, BM Snow, BR Dust/Haze.
    cell_h, cell_w = wh // 2, ww // 3
    clear = wx[:cell_h, :cell_w]
    rain = wx[:cell_h, 2 * cell_w :]

    target_h = 280
    panels = [
        _label_panel(_resize_h(narrow, target_h), "Narrow HD"),
        _label_panel(_resize_h(lowcost, target_h), "Low-Cost"),
        _label_panel(_resize_h(clear, target_h), "Clear day"),
        _label_panel(_resize_h(rain, target_h), "Rain"),
    ]
    # Equal height already; crop/pad to a common cell width for a clean strip.
    cell_w_tgt = min(p.shape[1] for p in panels)

    def _fit_w(img, width):
        if img.shape[1] == width:
            return img
        if img.shape[1] > width:
            x0 = (img.shape[1] - width) // 2
            return img[:, x0 : x0 + width]
        pad = np.zeros((img.shape[0], width - img.shape[1], 3), dtype=np.uint8)
        return np.hstack([img, pad])

    panels = [_fit_w(p, cell_w_tgt) for p in panels]
    gap = 3
    group_gap = 8
    white = np.full((target_h, gap, 3), 255, dtype=np.uint8)
    group = np.full((target_h, group_gap, 3), 255, dtype=np.uint8)
    # Cameras | Weather, with a wider gap between groups.
    strip = np.hstack(
        [panels[0], white, panels[1], group, panels[2], white, panels[3]]
    )

    fig, ax = plt.subplots(figsize=(6.6, 1.45), dpi=220)
    ax.imshow(strip)
    ax.axis("off")
    # Group boundary between camera pair and weather pair.
    split = 2 * cell_w_tgt + gap + group_gap // 2
    ax.axvline(split, color="#888888", linewidth=1.0, alpha=0.85)
    fig.tight_layout(pad=0.02)
    fig.savefig(outfile.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(outfile.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("Wrote", outfile.with_suffix(".pdf"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cross-val", action="store_true")
    ap.add_argument("--wsc26-only", action="store_true", help="Only build figures/wsc26_fig1 and fig2")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    FIGURES.mkdir(parents=True, exist_ok=True)

    if args.wsc26_only:
        plot_wsc26_composite(FIGURES / "wsc26_fig1")
        plot_wsc26_urban_panels(FIGURES / "wsc26_fig2")
        return

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
    plot_wsc26_composite(FIGURES / "wsc26_fig1")
    plot_wsc26_urban_panels(FIGURES / "wsc26_fig2")

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
