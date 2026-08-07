"""
Place Observer and intruder (e.g. Drone1) at a true Euclidean separation in *global* NED,
using per-vehicle local commands (moveToPositionAsync). AirSim local XY is offset by each
vehicle's spawn from Documents/AirSim/settings.json — see experiment_controller._to_local.

Usage (simulator running):
  python tools/place_drones_pair_ned.py --sep 10 --z -100 --camera-yaw-offset 180
  python tools/place_drones_pair_ned.py --sep 10 --z -100 --altitude-diff 2.5   # intruder NED Z +2.5 vs observer
  python tools/place_drones_pair_ned.py --sep 10 --z -100 --camera-yaw-offset 0   # if body already faces +X

Options:
  --observer-at 0 0      global NED XY for Observer (default 0 0)
  --bearing-deg 0        intruder direction from observer in NED XY (0 = +X north)
  --sep                  horizontal (XY) separation in metres between observer and intruder
  --altitude-diff        intruder Z minus observer Z in NED (+ = intruder more +Z, i.e. lower if Z+ is down)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import cosysairsim as airsim

REPO_ROOT = Path(__file__).resolve().parents[1]


def read_vehicle_spawns() -> dict[str, tuple[float, float, float]]:
    def _load(path: Path) -> dict[str, tuple[float, float, float]] | None:
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            settings = json.load(f)
        vehicles = settings.get("Vehicles", {})
        offsets = {}
        for v_name, v_cfg in vehicles.items():
            offsets[v_name] = (
                float(v_cfg.get("X", 0.0)),
                float(v_cfg.get("Y", 0.0)),
                float(v_cfg.get("Z", 0.0)),
            )
        return offsets

    home = Path(os.path.expanduser("~"))
    for settings_path in (
        home / "Documents" / "AirSim" / "settings.json",
        home / "OneDrive" / "Documents" / "AirSim" / "settings.json",
        home / "OneDrive - Personal" / "Documents" / "AirSim" / "settings.json",
    ):
        off = _load(settings_path)
        if off:
            print(f"Using settings: {settings_path}")
            return off
    for settings_path in (
        REPO_ROOT / "config" / "cosys_airsim_settings_extra_cameras.json",
        REPO_ROOT / "config" / "cosys_airsim_settings.json",
    ):
        off = _load(settings_path)
        if off:
            print(f"Using settings (repo fallback): {settings_path}")
            return off
    raise FileNotFoundError("No AirSim settings.json found (Documents or config/)")


def local_from_global(
    home: dict[str, tuple[float, float, float]], vehicle: str, gx: float, gy: float, gz: float
) -> tuple[float, float, float]:
    hx, hy, _ = home.get(vehicle, (0.0, 0.0, 0.0))
    return (gx - hx, gy - hy, gz)


def global_from_local(
    home: dict[str, tuple[float, float, float]], vehicle: str, c: airsim.MultirotorClient
) -> tuple[float, float, float]:
    p = c.simGetVehiclePose(vehicle_name=vehicle).position
    hx, hy, _ = home.get(vehicle, (0.0, 0.0, 0.0))
    return (p.x_val + hx, p.y_val + hy, p.z_val)


def run_place(
    *,
    ip: str = "127.0.0.1",
    observer: str = "Observer",
    intruder: str = "Drone1",
    sep: float = 10.0,
    z_observer: float = -100.0,
    altitude_diff: float = 0.0,
    z_intruder: float | None = None,
    observer_at: tuple[float, float] = (0.0, 0.0),
    bearing_deg: float = 0.0,
    camera_yaw_offset: float = 180.0,
    velocity: float = 6.0,
) -> None:
    """Programmatic API; same behaviour as CLI."""
    home = read_vehicle_spawns()
    for name, off in home.items():
        print(f"  spawn {name}: ({off[0]:.1f}, {off[1]:.1f}, {off[2]:.1f})")

    ox, oy = observer_at
    oz = float(z_observer)
    iz = float(z_intruder) if z_intruder is not None else (oz + float(altitude_diff))
    br = math.radians(bearing_deg)
    ix = ox + sep * math.cos(br)
    iy = oy + sep * math.sin(br)

    lo = local_from_global(home, observer, ox, oy, oz)
    li = local_from_global(home, intruder, ix, iy, iz)
    print(f"Target global NED: {observer} ({ox:.2f},{oy:.2f},{oz:.2f}) | {intruder} ({ix:.2f},{iy:.2f},{iz:.2f})")
    print(f"  altitude_diff (intruder Z - observer Z): {iz - oz:.2f} m")
    print(f"Local moveTo: {observer} {lo} | {intruder} {li}")

    c = airsim.MultirotorClient(ip=ip)
    c.confirmConnection()
    for v in (observer, intruder):
        c.enableApiControl(True, vehicle_name=v)
        c.armDisarm(True, vehicle_name=v)
    c.takeoffAsync(vehicle_name=observer).join()
    c.takeoffAsync(vehicle_name=intruder).join()

    c.moveToPositionAsync(lo[0], lo[1], lo[2], velocity, vehicle_name=observer).join()
    c.moveToPositionAsync(li[0], li[1], li[2], velocity, vehicle_name=intruder).join()
    time.sleep(1.0)

    def horiz_and_vert() -> tuple[tuple[float, float, float], tuple[float, float, float], float, float]:
        a = global_from_local(home, observer, c)
        b = global_from_local(home, intruder, c)
        dh = math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
        dv = b[2] - a[2]
        return a, b, dh, dv

    for _ in range(14):
        g1, g2, dh, _dv = horiz_and_vert()
        err = sep - dh
        if abs(err) < 0.08:
            break
        dx_, dy_ = g2[0] - g1[0], g2[1] - g1[1]
        hlen = max(1e-6, math.sqrt(dx_ * dx_ + dy_ * dy_))
        ux, uy = dx_ / hlen, dy_ / hlen
        g2_new_xy = (g2[0] + ux * err * 0.9, g2[1] + uy * err * 0.9)
        li2 = local_from_global(home, intruder, g2_new_xy[0], g2_new_xy[1], iz)
        c.moveToPositionAsync(li2[0], li2[1], li2[2], 4.0, vehicle_name=intruder).join()
        time.sleep(0.35)

    g1, g2, dh, dv_act = horiz_and_vert()
    d3 = math.sqrt((g1[0] - g2[0]) ** 2 + (g1[1] - g2[1]) ** 2 + (g1[2] - g2[2]) ** 2)
    print(f"Global after move: {observer} ({g1[0]:.2f},{g1[1]:.2f},{g1[2]:.2f})")
    print(f"Global after move: {intruder} ({g2[0]:.2f},{g2[1]:.2f},{g2[2]:.2f})")
    print(f"Horizontal sep (m): {dh:.2f} | dZ intruder-observer (m): {dv_act:.2f} | 3D (m): {d3:.2f}")

    dx, dy = g2[0] - g1[0], g2[1] - g1[1]
    bearing = math.degrees(math.atan2(dy, dx)) % 360.0
    yaw_cmd = (bearing + camera_yaw_offset) % 360.0
    if hasattr(c, "rotateToYawAsync"):
        c.rotateToYawAsync(yaw_cmd, timeout_sec=20, margin=2, vehicle_name=observer).join()
    print(
        f"Bearing OBS to INTR (deg): {bearing:.1f} | rotateToYawAsync: {yaw_cmd:.1f} "
        f"(offset {camera_yaw_offset:+.1f})"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="127.0.0.1")
    ap.add_argument("--observer", default="Observer")
    ap.add_argument("--intruder", default="Drone1")
    ap.add_argument(
        "--sep",
        type=float,
        default=10.0,
        help="Horizontal (XY) separation in metres (NED ground plane)",
    )
    ap.add_argument(
        "--z",
        type=float,
        default=-100.0,
        dest="z_observer",
        help="Observer global NED Z (negative = up in typical AirSim NED)",
    )
    ap.add_argument(
        "--altitude-diff",
        type=float,
        default=0.0,
        help="Intruder Z minus observer Z (NED). Positive = intruder more +Z than observer",
    )
    ap.add_argument(
        "--z-intruder",
        type=float,
        default=None,
        help="Override intruder global NED Z (if set, ignores --altitude-diff)",
    )
    ap.add_argument("--observer-at", nargs=2, type=float, default=(0.0, 0.0), metavar=("X", "Y"))
    ap.add_argument(
        "--bearing-deg",
        type=float,
        default=0.0,
        help="Intruder direction from observer in NED XY (deg, 0 = +X)",
    )
    ap.add_argument(
        "--camera-yaw-offset",
        type=float,
        default=180.0,
        help="Added to bearing for rotateToYawAsync (180 if camera faces opposite body +X)",
    )
    ap.add_argument("--velocity", type=float, default=6.0)
    args = ap.parse_args()

    run_place(
        ip=args.ip,
        observer=args.observer,
        intruder=args.intruder,
        sep=args.sep,
        z_observer=args.z_observer,
        altitude_diff=args.altitude_diff,
        z_intruder=args.z_intruder,
        observer_at=tuple(args.observer_at),
        bearing_deg=args.bearing_deg,
        camera_yaw_offset=args.camera_yaw_offset,
        velocity=args.velocity,
    )


if __name__ == "__main__":
    main()
