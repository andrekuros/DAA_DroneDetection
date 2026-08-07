"""
capture_gnss_scene.py — Foto do cenario urbano para o artigo
=============================================================
Captura a cena do Cosys-AirSim pela camera de terceira pessoa do veiculo PX4,
para a figura que mostra o ambiente de simulacao no extended abstract.

Nao interfere no voo: so le imagens, nao comanda nada. Pode rodar com o
experimento em andamento (o AirSim aceita multiplos clientes RPC).

Dependencias:
    pip install cosysairsim opencv-python numpy

Uso:
    # Uma foto agora
    python tools/capture_gnss_scene.py

    # Varias, espacadas, para escolher a melhor
    python tools/capture_gnss_scene.py --shots 6 --interval-s 4

    # Nome e pasta proprios
    python tools/capture_gnss_scene.py --out figures/wsc26_scene --camera scene_cam
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"

try:
    import cosysairsim as airsim
    HAS_AIRSIM = True
except ImportError:
    try:
        import airsim
        HAS_AIRSIM = True
    except ImportError:
        HAS_AIRSIM = False


def capture(client, camera: str, vehicle: str) -> np.ndarray | None:
    """Uma imagem RGB da camera indicada, como array BGR para o OpenCV."""
    responses = client.simGetImages(
        [airsim.ImageRequest(camera, airsim.ImageType.Scene, False, False)],
        vehicle_name=vehicle,
    )
    if not responses:
        return None
    r = responses[0]
    if r.height == 0 or r.width == 0:
        return None
    buf = np.frombuffer(r.image_data_uint8, dtype=np.uint8)
    return buf.reshape(r.height, r.width, 3)


def main() -> None:
    ap = argparse.ArgumentParser(description="Captura a cena urbana para a figura do artigo")
    ap.add_argument("--out", type=Path, default=FIGURES / "wsc26_gnss_scene",
                    help="Prefixo de saida (sem extensao)")
    ap.add_argument("--camera", default="scene_cam", metavar="NOME")
    ap.add_argument("--vehicle", default="PX4Drone", metavar="NOME")
    ap.add_argument("--ip", default="127.0.0.1", metavar="IP")
    ap.add_argument("--shots", type=int, default=1, metavar="N")
    ap.add_argument("--interval-s", type=float, default=3.0, metavar="S")
    args = ap.parse_args()

    if not HAS_AIRSIM:
        ap.error("cosysairsim nao instalado (pip install cosysairsim)")

    client = airsim.MultirotorClient(ip=args.ip)
    client.confirmConnection()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    saved = 0
    for i in range(args.shots):
        img = capture(client, args.camera, args.vehicle)
        if img is None:
            print(f"[AVISO] Camera '{args.camera}' nao retornou imagem. "
                  "Confira o bloco Cameras do settings.json e reinicie o simulador.")
            break
        suffix = "" if args.shots == 1 else f"_{i + 1:02d}"
        path = args.out.with_name(args.out.name + suffix).with_suffix(".png")
        cv2.imwrite(str(path), img)
        k = client.simGetGroundTruthKinematics(vehicle_name=args.vehicle)
        print(f"  -> {path.name}  ({img.shape[1]}x{img.shape[0]}) "
              f"X={k.position.x_val:.0f} alt={-k.position.z_val:.0f}m")
        saved += 1
        if i < args.shots - 1:
            time.sleep(args.interval_s)

    print(f"[OK] {saved} imagem(ns) em {args.out.parent}")


if __name__ == "__main__":
    main()
