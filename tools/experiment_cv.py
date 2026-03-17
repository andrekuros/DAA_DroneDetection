"""
experiment_cv.py — Experimentos simples no modo ComputerVision do Cosys-AirSim
==============================================================================
Gera uma sequência de imagens usando apenas a API de Computer Vision
(`SimMode: "ComputerVision"`), sem física de drone.

Ideia: câmera (observador) fixa em certa posição e um "drone intruso
virtual" se aproximando em linha reta. Na prática, movemos apenas a
câmera ao longo da trajetória e salvamos as imagens.

Uso típico:

    # 1) Copiar o settings de ComputerVision
    #    (faz isso uma vez antes de abrir o Unreal)
    Copy-Item config\\cosys_airsim_computervision.json `
        \"$env:USERPROFILE\\Documents\\AirSim\\settings.json\"

    # 2) Abrir o projeto Unreal/Cosys-AirSim (Blocks ou outro mapa)

    # 3) Gerar uma aproximação frontal simples (RGB)
    python tools\\experiment_cv.py ^
        --frames 200 ^
        --start-dist 200 ^
        --end-dist 10 ^
        --altitude 50 ^
        --output-dir dataset_cv\\frontal

Todas as distâncias são em metros no sistema NED (X = Norte, Y = Leste,
Z para baixo; Z negativo = acima do solo).
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
import csv

try:
    import cosysairsim as airsim
except ImportError:
    import airsim  # type: ignore


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def run_cv_experiment(
    frames: int,
    start_dist: float,
    end_dist: float,
    altitude: float,
    azimuth_deg: float,
    output_dir: Path,
    sleep_s: float,
) -> None:
    """
    Move a câmera ao longo de uma linha reta em direção ao "observador".

    Convenção:
        - Observador lógico no (0, 0, -altitude)
        - "Intruso" virtual começa em (dist, 0, -altitude) no eixo X (frontal)
        - A câmera percorre essa linha do ponto inicial ao final.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_rows: list[dict] = []

    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)

    # Em modo ComputerVision, a câmera "0" fica no veículo invisível.
    camera_name = "0"

    # Converte azimute para deslocamento inicial em NED
    az_rad = math.radians(azimuth_deg)
    start_x = start_dist * math.cos(az_rad)
    start_y = start_dist * math.sin(az_rad)
    end_x = end_dist * math.cos(az_rad)
    end_y = end_dist * math.sin(az_rad)
    z = -altitude  # NED: negativo = acima do solo

    print(
        f"[INFO] Modo ComputerVision | frames={frames} | "
        f"start=({start_x:.1f},{start_y:.1f},{z:.1f}) -> "
        f"end=({end_x:.1f},{end_y:.1f},{z:.1f})"
    )

    for i in range(frames):
        t = i / max(frames - 1, 1)
        x = lerp(start_x, end_x, t)
        y = lerp(start_y, end_y, t)

        pose = airsim.Pose(
            position_val=airsim.Vector3r(x, y, z),
            orientation_val=airsim.euler_to_quaternion(0, 0, 0),
        )
        client.simSetVehiclePose(pose, ignore_collision=True)

        # Captura apenas cena RGB (ImageType.Scene)
        responses = client.simGetImages(
            [airsim.ImageRequest(camera_name, airsim.ImageType.Scene, compress=True)]
        )
        if not responses:
            print(f"[AVISO] Nenhuma resposta de imagem no frame {i}")
            continue
        resp = responses[0]

        if len(resp.image_data_uint8) == 0:
            print(f"[AVISO] Imagem vazia no frame {i}")
            continue

        out_path = output_dir / f"frame_{i:06d}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        airsim.write_file(str(out_path), resp.image_data_uint8)

        # Metadados por frame
        # Observador logico em (0,0,-altitude); aqui usamos distancia 3D
        dist = math.sqrt(x * x + y * y + (z + altitude) ** 2)
        # Orientacao da camera neste exemplo: sempre roll=pitch=yaw=0
        roll = 0.0
        pitch = 0.0
        yaw = 0.0
        meta_rows.append(
            {
                "type": "camera",
                "frame": i,
                "t_norm": float(t),
                "cam_x_m": float(x),
                "cam_y_m": float(y),
                "cam_z_m": float(z),
                "cam_roll_rad": float(roll),
                "cam_pitch_rad": float(pitch),
                "cam_yaw_rad": float(yaw),
                "observer_x_m": 0.0,
                "observer_y_m": 0.0,
                "observer_z_m": -float(altitude),
                "distance_m": float(dist),
                "azimuth_deg": float(azimuth_deg),
                "altitude_m": float(altitude),
            }
        )

        if i % 20 == 0:
            print(
                f"  frame {i:05d} | "
                f"dist={dist:6.1f}m | "
                f"z_cam={z:6.1f}m | "
                f"z_obs={-altitude:6.1f}m | "
                f"pos=({x:6.1f},{y:6.1f},{z:6.1f}) | "
                f"salvo: {out_path.name}"
            )

        if sleep_s > 0:
            time.sleep(sleep_s)

    # Salva metadados em CSV
    if meta_rows:
        csv_path = output_dir / "metadata.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(meta_rows[0].keys()))
            writer.writeheader()
            writer.writerows(meta_rows)
        print(f"[CONCLUIDO] {len(meta_rows)} frames | metadados: {csv_path}")
    else:
        print(f"[CONCLUIDO] Nenhum frame valido gerado em {output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Experimentos simples em modo ComputerVision (Cosys-AirSim)."
    )
    p.add_argument("--frames", type=int, default=200, help="Numero de frames (padrao: 200).")
    p.add_argument(
        "--start-dist",
        type=float,
        default=200.0,
        help="Distancia inicial do intruso virtual (m, padrao: 200).",
    )
    p.add_argument(
        "--end-dist",
        type=float,
        default=10.0,
        help="Distancia final (m) em relacao ao observador (padrao: 10).",
    )
    p.add_argument(
        "--altitude",
        type=float,
        default=50.0,
        help="Altitude (m) acima do solo (padrao: 50).",
    )
    p.add_argument(
        "--azimuth",
        type=float,
        default=0.0,
        help="Azimute em graus (0=Norte/frontal, 90=Leste/direita, ...).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset_cv") / "frontal",
        help="Diretorio de saida (padrao: dataset_cv/frontal).",
    )
    p.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Pausa em segundos entre frames (0 = o mais rapido possivel).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_cv_experiment(
        frames=args.frames,
        start_dist=args.start_dist,
        end_dist=args.end_dist,
        altitude=args.altitude,
        azimuth_deg=args.azimuth,
        output_dir=args.output_dir,
        sleep_s=args.sleep,
    )


if __name__ == "__main__":
    main()

