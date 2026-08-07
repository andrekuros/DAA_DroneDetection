"""
Capture a synchronized RGB frame from all 4 Observer cameras in a running
Cosys-AirSim instance, run the trained YOLOv8s model for drone detection,
and build a 2x2 composite image suitable for the report.

Requirements:
    - Cosys-AirSim running with config/cosys_airsim_settings_extra_cameras.json
    - Trained model at runs/detect/runs/train/exp30_multicam_yolov8s/weights/best.pt

Usage:
    python tools/capture_multicam_composite.py [--ip 127.0.0.1] [--out figures/new_composite.png] [--conf 0.25]
    python tools/capture_multicam_composite.py --place --place-sep 10 --place-z -100 --place-altitude-diff 0 --conf 0.8
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

CAMERAS = [
    ("front_center",      "Standard Forward"),
    ("front_narrow_hd",   "Narrow HD (Long-Range)"),
    ("front_medium",      "Medium FOV"),
    ("front_lowcost_640", "Low-Cost Wide (640 px)"),
]

MODEL_PATH = ROOT / "runs" / "detect" / "runs" / "train" / "exp30_multicam_yolov8s" / "weights" / "best.pt"
OBSERVER_VEHICLE = "Observer"


def capture_rgb(client, camera_name: str) -> np.ndarray | None:
    """Capture a single RGB frame from the specified camera on the Observer."""
    import cosysairsim as airsim
    responses = client.simGetImages(
        [airsim.ImageRequest(camera_name, airsim.ImageType.Scene, compress=True)],
        vehicle_name=OBSERVER_VEHICLE,
    )
    if not responses or len(responses[0].image_data_uint8) == 0:
        return None
    buf = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img


def run_yolo(model, img: np.ndarray, conf: float = 0.25) -> list[dict]:
    """Run YOLO inference and return list of detections."""
    results = model.predict(img, conf=conf, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            detections.append({
                "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                "conf": float(box.conf[0]),
                "cls": int(box.cls[0]),
            })
    return detections


def draw_detections(img: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Draw bounding boxes and confidence on the image."""
    out = img.copy()
    for d in detections:
        thick = max(2, min(out.shape[:2]) // 250)
        cv2.rectangle(out, (d["x1"], d["y1"]), (d["x2"], d["y2"]),
                      (0, 255, 0), thick, lineType=cv2.LINE_AA)
        label = f"drone {d['conf']:.0%}"
        font_scale = max(0.5, min(out.shape[:2]) / 1400)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)
        cv2.rectangle(out, (d["x1"], d["y1"] - th - 8), (d["x1"] + tw + 4, d["y1"]),
                      (0, 255, 0), -1)
        cv2.putText(out, label, (d["x1"] + 2, d["y1"] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thick, cv2.LINE_AA)
    return out


def add_camera_label(img: np.ndarray, label: str) -> np.ndarray:
    """Overlay the camera name label on the image."""
    out = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.6, min(out.shape[:2]) / 1200)
    thick = max(2, int(scale * 3))
    cv2.putText(out, label, (12, 32), font, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(out, label, (12, 32), font, scale, (0, 200, 255), thick, cv2.LINE_AA)
    return out


def build_composite(panels: list[np.ndarray], target_cell_w: int = 640) -> np.ndarray:
    """Resize panels and arrange in 2x2 grid."""
    resized = []
    for p in panels:
        h, w = p.shape[:2]
        scale = target_cell_w / w
        resized.append(cv2.resize(p, (target_cell_w, int(h * scale)), interpolation=cv2.INTER_AREA))

    max_h = max(r.shape[0] for r in resized)
    padded = []
    for r in resized:
        if r.shape[0] < max_h:
            pad = np.zeros((max_h - r.shape[0], r.shape[1], 3), dtype=np.uint8)
            r = np.vstack([r, pad])
        padded.append(r)

    n = len(padded)
    top = np.hstack(padded[:2])
    bot = np.hstack(padded[2:4]) if n >= 4 else np.hstack(padded[2:]) if n > 2 else np.zeros_like(top)
    if top.shape[1] != bot.shape[1]:
        tw = max(top.shape[1], bot.shape[1])
        if top.shape[1] < tw:
            top = np.hstack([top, np.zeros((top.shape[0], tw - top.shape[1], 3), dtype=np.uint8)])
        if bot.shape[1] < tw:
            bot = np.hstack([bot, np.zeros((bot.shape[0], tw - bot.shape[1], 3), dtype=np.uint8)])
    return np.vstack([top, bot])


def main():
    ap = argparse.ArgumentParser(
        description="Multicam RGB + optional YOLO boxes + 2x2 composite. "
        "Use --place to move Observer/Drone1 before capture (see place_drones_pair_ned)."
    )
    ap.add_argument("--ip", default="127.0.0.1")
    ap.add_argument("--out", type=Path, default=ROOT / "figures" / "exp30_multicam_newenv.png")
    ap.add_argument(
        "--conf",
        "--yolo-conf",
        "--yolo-threshold",
        type=float,
        default=0.25,
        dest="conf",
        metavar="P",
        help="YOLO minimum confidence in [0,1]; detections below this are not drawn (default 0.25)",
    )
    ap.add_argument("--no-yolo", action="store_true", help="Skip YOLO inference, just capture raw frames")
    ap.add_argument("--save-individual", action="store_true", help="Also save each camera frame separately")
    ap.add_argument(
        "--place",
        action="store_true",
        help="Run place_drones_pair_ned before capture (AirSim must be running)",
    )
    ap.add_argument("--place-sep", type=float, default=10.0, help="Horizontal XY separation (m)")
    ap.add_argument("--place-z", type=float, default=-100.0, help="Observer global NED Z")
    ap.add_argument(
        "--place-altitude-diff",
        type=float,
        default=0.0,
        help="Intruder NED Z minus observer NED Z (same as place_drones_pair_ned --altitude-diff)",
    )
    ap.add_argument(
        "--place-z-intruder",
        type=float,
        default=None,
        help="Optional intruder global NED Z (overrides --place-altitude-diff)",
    )
    ap.add_argument("--place-yaw-offset", type=float, default=180.0, help="rotateToYawAsync offset (deg)")
    ap.add_argument("--place-bearing-deg", type=float, default=0.0, help="Intruder direction from observer in XY")
    args = ap.parse_args()

    if args.place:
        tools_dir = Path(__file__).resolve().parent
        if str(tools_dir) not in sys.path:
            sys.path.insert(0, str(tools_dir))
        from place_drones_pair_ned import run_place

        print("Placing drones (--place)...")
        run_place(
            ip=args.ip,
            sep=args.place_sep,
            z_observer=args.place_z,
            altitude_diff=args.place_altitude_diff,
            z_intruder=args.place_z_intruder,
            bearing_deg=args.place_bearing_deg,
            camera_yaw_offset=args.place_yaw_offset,
        )
        time.sleep(0.5)

    print("Connecting to Cosys-AirSim...")
    import cosysairsim as airsim
    client = airsim.MultirotorClient(ip=args.ip)
    client.confirmConnection()
    print("Connected.")
    if not args.no_yolo:
        print(f"YOLO confidence threshold: {args.conf}")

    model = None
    if not args.no_yolo:
        if MODEL_PATH.exists():
            from ultralytics import YOLO
            model = YOLO(str(MODEL_PATH))
            print(f"YOLO model loaded: {MODEL_PATH.name}")
        else:
            print(f"[WARN] Model not found at {MODEL_PATH}, skipping detection")

    time.sleep(0.5)

    panels = []
    for cam_id, cam_label in CAMERAS:
        print(f"  Capturing {cam_id}...")
        img = capture_rgb(client, cam_id)
        if img is None:
            print(f"    [SKIP] Empty response from {cam_id}")
            continue

        if args.save_individual:
            ind_path = args.out.parent / f"{args.out.stem}_{cam_id}.png"
            cv2.imwrite(str(ind_path), img)
            print(f"    Saved individual: {ind_path.name}")

        if model is not None:
            dets = run_yolo(model, img, conf=args.conf)
            print(f"    {len(dets)} detection(s)")
            img = draw_detections(img, dets)

        img = add_camera_label(img, cam_label)
        panels.append(img)

    if len(panels) < 2:
        print("[ERROR] Not enough camera frames captured. Is AirSim running with extra_cameras settings?")
        sys.exit(1)

    composite = build_composite(panels)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), composite, [cv2.IMWRITE_PNG_COMPRESSION, 5])
    print(f"\nComposite saved: {args.out} ({composite.shape[1]}x{composite.shape[0]})")


if __name__ == "__main__":
    main()
