"""
airsim_gt.py — Leitura de ground truth do Cosys-AirSim
======================================================
Fornece a posicao VERDADEIRA do veiculo no simulador, para comparar contra a
estimativa do EKF2 do PX4 (ver px4_link.py).

Diferenca importante em relacao ao resto do repo: experiment_controller.py usa
`getMultirotorState().kinematics_estimated`, que com SimpleFlight coincide com a
verdade. Com PX4 no circuito isso deixa de valer — `kinematics_estimated` passa a
refletir o estimador. Aqui usamos `simGetGroundTruthKinematics()`, que e a pose
real do ator no Unreal, independente de qualquer estimador.

Dependencias:
    pip install cosysairsim

Uso (normalmente importado; standalone serve de smoke test):
    python tools/airsim_gt.py --check --vehicle PX4Drone

Coordenadas NED: X=Norte, Y=Leste, Z=para baixo.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ─── Importacao Cosys-AirSim (permite --dry-run sem simulador) ───────────────
try:
    import cosysairsim as airsim
    HAS_AIRSIM = True
    if not hasattr(airsim, "to_quaternion"):
        airsim.to_quaternion = airsim.euler_to_quaternion
    if not hasattr(airsim, "to_eularian_angles"):
        airsim.to_eularian_angles = airsim.quaternion_to_euler_angles
except ImportError:
    try:
        import airsim
        HAS_AIRSIM = True
    except ImportError:
        HAS_AIRSIM = False
        print("[AVISO] cosysairsim nao instalado. Apenas --dry-run disponivel.")


# Mesmos caminhos que experiment_controller._read_vehicle_spawns() sonda.
_SETTINGS_CANDIDATES = [
    Path.home() / "Documents" / "AirSim" / "settings.json",
    Path.home() / "OneDrive" / "Documents" / "AirSim" / "settings.json",
    Path.home() / "OneDrive - Personal" / "Documents" / "AirSim" / "settings.json",
    ROOT / "config" / "cosys_airsim_px4_settings.json",
]

# Retornos mais proximos que isto sao ecos na propria fuselagem/helices, nao cena.
SELF_HIT_M = 2.0


def read_vehicle_spawns() -> dict[str, tuple[float, float, float]]:
    """
    Le os offsets de spawn (X,Y,Z) por veiculo do settings.json ativo.

    Necessario porque simGetGroundTruthKinematics/simGetVehiclePose retornam
    coordenadas LOCAIS ao spawn do veiculo no Cosys-AirSim; a posicao global e
    local + offset. Mesma logica de experiment_controller._read_vehicle_spawns().
    """
    for path in _SETTINGS_CANDIDATES:
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            continue
        spawns: dict[str, tuple[float, float, float]] = {}
        for name, v in (data.get("Vehicles") or {}).items():
            spawns[name] = (
                float(v.get("X", 0.0)),
                float(v.get("Y", 0.0)),
                float(v.get("Z", 0.0)),
            )
        if spawns:
            return spawns
    return {}


@dataclass
class SensorSample:
    """
    Medidas cruas dos sensores durante a janela sem GNSS.

    Sao exatamente as entradas que um factor graph consome: IMU para fatores de
    pre-integracao, barometro para altitude, magnetometro para heading, e a
    densidade de estrutura do LiDAR para saber quando fatores de scan-matching
    teriam sinal. Gravar isso agora permite reprocessar a mesma janela offline e
    comparar o factor graph contra o EKF2 sobre o dado identico.
    """
    # IMU
    ax: float = float("nan")
    ay: float = float("nan")
    az: float = float("nan")
    gx: float = float("nan")
    gy: float = float("nan")
    gz: float = float("nan")
    # Barometro / magnetometro
    baro_alt_m: float = float("nan")
    baro_pressure: float = float("nan")
    mag_x: float = float("nan")
    mag_y: float = float("nan")
    mag_z: float = float("nan")
    # Estrutura visivel ao LiDAR (proxy de observabilidade para scan-matching)
    lidar_points: int = -1
    lidar_mean_range_m: float = float("nan")
    lidar_min_range_m: float = float("nan")
    # Alcance mediano do cone nadir: com o sensor apontado para baixo isto e a
    # altura sobre o que esta embaixo (telhado ou rua), nao a altitude absoluta.
    # E uma observacao independente de barometro e GNSS, entao entra no factor
    # graph como fator de altura sempre que ha retorno.
    lidar_agl_m: float = float("nan")
    # Colisao: numa cena urbana densa o veiculo bate em predios e props. Um impacto
    # perturba a dinamica e a IMU, contaminando justamente a deriva que medimos —
    # entao o trecho afetado precisa ser identificavel no CSV, nao descoberto depois.
    collided: int = 0
    collision_object: str = ""


@dataclass
class GroundTruthSample:
    """Pose verdadeira do veiculo no simulador (NED global, metros)."""
    t_wall: float = 0.0
    x_m: float = float("nan")
    y_m: float = float("nan")
    z_m: float = float("nan")
    vx_ms: float = float("nan")
    vy_ms: float = float("nan")
    vz_ms: float = float("nan")
    roll_rad: float = float("nan")
    pitch_rad: float = float("nan")
    yaw_rad: float = float("nan")
    valid: bool = False


@dataclass
class AirSimGroundTruth:
    """Cliente sincrono do Cosys-AirSim, restrito a leitura de ground truth."""

    vehicle_name: str = "PX4Drone"
    ip: str = "127.0.0.1"
    lidar_name: str = "LidarNadir"
    client: object | None = field(default=None, init=False)
    spawn_offset: tuple[float, float, float] = field(default=(0.0, 0.0, 0.0), init=False)
    _lidar_ok: bool = field(default=True, init=False)
    _pool: object | None = field(default=None, init=False)
    _last_collision_ts: int = field(default=0, init=False)

    @staticmethod
    def _init_rpc_thread() -> None:
        """
        Da um event loop a thread do pool.

        O cliente do cosysairsim usa msgpack-rpc sobre tornado, que exige um event
        loop no thread corrente. Sem isto toda chamada morre com "There is no
        current event loop in thread ...".
        """
        import asyncio as _a
        try:
            _a.get_event_loop()
        except RuntimeError:
            _a.set_event_loop(_a.new_event_loop())

    def _ensure_pool(self):
        """
        Executor de UMA thread, dedicado ao RPC do AirSim.

        Duas razoes para nao usar asyncio.to_thread: o pool padrao espalha as
        chamadas por varias threads, e (a) o cliente RPC nao e thread-safe, (b)
        cada thread nova precisaria do event loop acima. Uma thread fixa resolve
        os dois de uma vez e serializa os RPCs, que e o comportamento correto.
        """
        if self._pool is None:
            from concurrent.futures import ThreadPoolExecutor
            self._pool = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="airsim-rpc",
                initializer=self._init_rpc_thread,
            )
        return self._pool

    def read_all(self, with_lidar: bool = True) -> tuple[GroundTruthSample, SensorSample]:
        """Ground truth + sensores numa unica ida ao executor (uma thread, RPCs em ordem)."""
        return self.get_ground_truth(), self.get_sensors(with_lidar)

    async def read_all_async(self, with_lidar: bool = True):
        """Versao aguardavel, sempre na thread dedicada."""
        import asyncio as _a
        loop = _a.get_running_loop()
        return await loop.run_in_executor(self._ensure_pool(), self.read_all, with_lidar)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None

    def connect(self) -> None:
        if not HAS_AIRSIM:
            raise RuntimeError("cosysairsim nao instalado (pip install cosysairsim).")
        self._init_rpc_thread()
        print(f"[INFO] Conectando ao Cosys-AirSim em {self.ip} ...")
        self.client = airsim.MultirotorClient(ip=self.ip)
        self.client.confirmConnection()

        spawns = read_vehicle_spawns()
        self.spawn_offset = spawns.get(self.vehicle_name, (0.0, 0.0, 0.0))
        print(f"[OK] Cosys-AirSim conectado. Spawn de '{self.vehicle_name}': {self.spawn_offset}")

        try:
            available = self.client.listVehicles()
            if available and self.vehicle_name not in available:
                print(
                    f"[AVISO] Veiculo '{self.vehicle_name}' nao esta em {available}. "
                    "Confira o settings.json copiado para Documents/AirSim."
                )
        except Exception:
            pass

        # Ancorar o timestamp de colisao no estado ATUAL da cena.
        #
        # `simGetCollisionInfo` guarda a ultima colisao de sempre, e o AirSim nao
        # a limpa entre voos. Cada voo roda em processo novo, entao sem esta ancora
        # `_last_collision_ts` comeca em 0, a primeira leitura ja difere, e o run e
        # marcado como colidido em t=0 com o impacto do voo ANTERIOR. Na campanha
        # isso vira cascata: todo retry nasce contaminado e queima o orcamento de
        # reposicao sem nunca produzir um voo limpo.
        try:
            ci = self.client.simGetCollisionInfo(vehicle_name=self.vehicle_name)
            self._last_collision_ts = int(getattr(ci, "time_stamp", 0) or 0)
        except Exception:
            self._last_collision_ts = 0

    def get_ground_truth(self) -> GroundTruthSample:
        """
        Pose real do veiculo. Chamada RPC BLOQUEANTE — no orquestrador assincrono
        ela roda dentro de asyncio.to_thread para nao travar o event loop do MAVSDK.
        """
        try:
            kin = self.client.simGetGroundTruthKinematics(vehicle_name=self.vehicle_name)
        except Exception as e:
            print(f"[AVISO] simGetGroundTruthKinematics falhou: {e}")
            return GroundTruthSample(t_wall=time.time(), valid=False)

        pos, vel = kin.position, kin.linear_velocity
        # Cosys-AirSim devolve (pitch, roll, yaw) nesta ordem.
        pitch, roll, yaw = airsim.to_eularian_angles(kin.orientation)
        hx, hy, hz = self.spawn_offset

        return GroundTruthSample(
            t_wall=time.time(),
            x_m=pos.x_val + hx,
            y_m=pos.y_val + hy,
            z_m=pos.z_val + hz,
            vx_ms=vel.x_val,
            vy_ms=vel.y_val,
            vz_ms=vel.z_val,
            roll_rad=roll,
            pitch_rad=pitch,
            yaw_rad=yaw,
            valid=True,
        )


    def get_sensors(self, with_lidar: bool = True) -> SensorSample:
        """
        Medidas cruas de IMU, barometro, magnetometro e densidade do LiDAR.

        Cada sensor e lido isoladamente: uma falha (sensor ausente no settings.json)
        degrada so aquele campo em vez de derrubar a amostra inteira. Bloqueante,
        como todo RPC do AirSim — chamar via asyncio.to_thread.
        """
        s = SensorSample()
        v = self.vehicle_name

        try:
            imu = self.client.getImuData(vehicle_name=v)
            s.ax, s.ay, s.az = (imu.linear_acceleration.x_val,
                                imu.linear_acceleration.y_val,
                                imu.linear_acceleration.z_val)
            s.gx, s.gy, s.gz = (imu.angular_velocity.x_val,
                                imu.angular_velocity.y_val,
                                imu.angular_velocity.z_val)
        except Exception:
            pass

        try:
            b = self.client.getBarometerData(vehicle_name=v)
            s.baro_alt_m, s.baro_pressure = b.altitude, b.pressure
        except Exception:
            pass

        try:
            m = self.client.getMagnetometerData(vehicle_name=v)
            s.mag_x, s.mag_y, s.mag_z = (m.magnetic_field_body.x_val,
                                         m.magnetic_field_body.y_val,
                                         m.magnetic_field_body.z_val)
        except Exception:
            pass

        # `has_collided` do AirSim e sticky: fica True para sempre depois do primeiro
        # contato, inclusive o encostao no solo/pedestre no spawn. Usar o flag cru
        # marcaria todo voo como contaminado. Detectamos por BORDA, comparando o
        # time_stamp da colisao com o da leitura anterior.
        try:
            ci = self.client.simGetCollisionInfo(vehicle_name=v)
            ts = int(getattr(ci, "time_stamp", 0) or 0)
            if ci.has_collided and ts and ts != self._last_collision_ts:
                self._last_collision_ts = ts
                s.collided = 1
                s.collision_object = str(ci.object_name)[:48]
        except Exception:
            pass

        # LiDAR e a leitura mais cara (nuvem inteira pelo RPC); o chamador decide
        # a cadencia e _lidar_ok evita insistir se o sensor nao existe.
        if with_lidar and self._lidar_ok:
            try:
                ld = self.client.getLidarData(lidar_name=self.lidar_name, vehicle_name=v)
                pts = ld.point_cloud
                n = len(pts) // 3
                # A nuvem vem com entradas nulas (raios sem retorno aparecem como
                # 0,0,0). Sem filtrar, a contagem satura no maximo do sensor e a
                # distancia media desaba — medimos 7,2 m de media voando a 60 m,
                # fisicamente impossivel. So pontos com retorno real interessam
                # como proxy de estrutura para scan-matching.
                # O limiar de 0,5 m era baixo demais: voando a 250 m com alcance de
                # 80 m o solo ficava fora de alcance e o que sobrava eram ecos na
                # propria fuselagem, a ~0,6 m. Media de 0,6 m nao e estrutura urbana,
                # e o proprio drone. Cortar em 2 m descarta esses ecos.
                rr = []
                for i in range(0, n * 3, 3):
                    d = math.sqrt(pts[i] ** 2 + pts[i + 1] ** 2 + pts[i + 2] ** 2)
                    if d > SELF_HIT_M:
                        rr.append(d)
                s.lidar_points = len(rr)
                if rr:
                    s.lidar_mean_range_m = sum(rr) / len(rr)
                    s.lidar_min_range_m = min(rr)
                    # Mediana em vez de media: o cone nadir cruza bordas de telhado,
                    # onde poucos raios caem na rua 100 m abaixo. A media desce com
                    # esses outliers; a mediana fica na superficie dominante.
                    rr.sort()
                    s.lidar_agl_m = rr[len(rr) // 2]
            except Exception as e:
                self._lidar_ok = False
                print(f"[AVISO] LiDAR '{self.lidar_name}' indisponivel ({e}); seguindo sem ele.")
        return s


class FrameRecorder:
    """
    Grava frames da camera nadir e nuvens do LiDAR durante o voo.

    Roda em thread propria com um cliente RPC proprio, e nao no executor do
    AirSimGroundTruth: capturar 1024x768 custa ordens de grandeza mais que ler
    IMU, e serializar as duas coisas na mesma thread furaria a cadencia de 20 Hz
    da telemetria. Cliente separado tambem evita compartilhar um socket msgpack
    que nao e thread-safe.

    Frames sao gravados em JPEG (q=90). O RPC entrega RGB cru — 2,4 MB por frame
    a 1024x768 — e guardar isso direto encheria o disco em um voo; o JPEG cai
    para ~100 kB sem prejudicar rastreio de features.

    O indice frames.csv associa cada arquivo ao relogio de parede (mesma base do
    telemetry.csv, o que permite juntar os dois depois) e ao ground truth do
    instante, necessario para avaliar o VIO contra a verdade.
    """

    INDEX_FIELDS = [
        "t_wall", "frame_file", "cloud_file",
        "gt_x", "gt_y", "gt_z", "gt_roll", "gt_pitch", "gt_yaw",
    ]

    def __init__(
        self,
        out_dir: Path,
        vehicle_name: str = "PX4Drone",
        ip: str = "127.0.0.1",
        camera: str = "vio_cam",
        lidar_name: str = "LidarNadir",
        frame_hz: float = 4.0,
        cloud_hz: float = 2.0,
        jpeg_quality: int = 90,
    ):
        self.out_dir = Path(out_dir)
        self.frames_dir = self.out_dir / "frames"
        self.clouds_dir = self.out_dir / "clouds"
        self.vehicle_name = vehicle_name
        self.ip = ip
        self.camera = camera
        self.lidar_name = lidar_name
        self.frame_hz = frame_hz
        self.cloud_hz = cloud_hz
        self.jpeg_quality = int(jpeg_quality)
        self.n_frames = 0
        self.n_clouds = 0
        self.errors = 0
        self._stop = None
        self._thread = None

    def start(self) -> None:
        import threading

        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.clouds_dir.mkdir(parents=True, exist_ok=True)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="frame-rec", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._stop is not None:
            self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None

    def _run(self) -> None:
        import csv as _csv

        import cv2
        import numpy as np

        AirSimGroundTruth._init_rpc_thread()
        try:
            client = airsim.MultirotorClient(ip=self.ip)
            client.confirmConnection()
        except Exception as e:
            print(f"[AVISO] FrameRecorder nao conectou ({e}); voo segue sem frames.")
            return

        idx_fh = (self.out_dir / "frames.csv").open("w", newline="", encoding="utf-8")
        idx = _csv.DictWriter(idx_fh, fieldnames=self.INDEX_FIELDS)
        idx.writeheader()

        period = 1.0 / max(self.frame_hz, 0.1)
        cloud_every = max(1, int(round(self.frame_hz / max(self.cloud_hz, 0.1))))
        enc = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        tick = 0
        next_t = time.monotonic()

        while not self._stop.is_set():
            t_wall = time.time()
            frame_file = cloud_file = ""

            try:
                resp = client.simGetImages(
                    [airsim.ImageRequest(self.camera, airsim.ImageType.Scene, False, False)],
                    vehicle_name=self.vehicle_name,
                )
                r = resp[0] if resp else None
                if r is not None and r.height > 0 and len(r.image_data_uint8) > 0:
                    buf = np.frombuffer(r.image_data_uint8, dtype=np.uint8)
                    ch = buf.size // (r.height * r.width)
                    img = buf.reshape(r.height, r.width, ch)[:, :, :3]
                    frame_file = f"frames/f_{self.n_frames:06d}.jpg"
                    cv2.imwrite(str(self.out_dir / frame_file), img, enc)
                    self.n_frames += 1
            except Exception:
                self.errors += 1

            if (tick % cloud_every) == 0:
                try:
                    ld = client.getLidarData(lidar_name=self.lidar_name,
                                             vehicle_name=self.vehicle_name)
                    pts = np.asarray(ld.point_cloud, dtype=np.float32)
                    if pts.size >= 3:
                        pts = pts.reshape(-1, 3)
                        # Descarta raios sem retorno (0,0,0) e ecos na fuselagem antes
                        # de gravar: a nuvem crua e majoritariamente zeros e triplicaria
                        # o arquivo sem informacao.
                        d = np.linalg.norm(pts, axis=1)
                        pts = pts[d > SELF_HIT_M]
                        if pts.size:
                            cloud_file = f"clouds/c_{self.n_clouds:06d}.npy"
                            np.save(self.out_dir / cloud_file, pts)
                            self.n_clouds += 1
                except Exception:
                    self.errors += 1

            row = {k: "" for k in self.INDEX_FIELDS}
            row["t_wall"] = f"{t_wall:.4f}"
            row["frame_file"] = frame_file
            row["cloud_file"] = cloud_file
            try:
                k = client.simGetGroundTruthKinematics(vehicle_name=self.vehicle_name)
                p, o = k.position, k.orientation
                roll, pitch, yaw = airsim.to_eularian_angles(o)
                row.update(gt_x=f"{p.x_val:.4f}", gt_y=f"{p.y_val:.4f}", gt_z=f"{p.z_val:.4f}",
                           gt_roll=f"{roll:.6f}", gt_pitch=f"{pitch:.6f}", gt_yaw=f"{yaw:.6f}")
            except Exception:
                self.errors += 1
            idx.writerow(row)
            idx_fh.flush()

            tick += 1
            next_t += period
            sleep_s = next_t - time.monotonic()
            if sleep_s > 0:
                self._stop.wait(sleep_s)
            else:
                # Captura mais lenta que a cadencia pedida: reancorar em vez de
                # acumular atraso, senao o laco entra em corrida sem folga.
                next_t = time.monotonic()

        idx_fh.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Smoke test da leitura de ground truth do Cosys-AirSim")
    ap.add_argument("--check", action="store_true", help="Conecta e imprime algumas amostras")
    ap.add_argument("--vehicle", default="PX4Drone", metavar="NOME")
    ap.add_argument("--ip", default="127.0.0.1", metavar="IP")
    ap.add_argument("--samples", type=int, default=10, metavar="N")
    args = ap.parse_args()

    if not args.check:
        ap.error("use --check (este modulo e normalmente importado, nao executado)")

    gt = AirSimGroundTruth(vehicle_name=args.vehicle, ip=args.ip)
    gt.connect()
    for _ in range(args.samples):
        s = gt.get_ground_truth()
        sen = gt.get_sensors()
        print(f"  X={s.x_m:8.2f} Y={s.y_m:8.2f} Z={s.z_m:8.2f} | "
              f"az={sen.az:6.2f} baro={sen.baro_alt_m:7.2f} "
              f"lidar_pts={sen.lidar_points:5d} mean_r={sen.lidar_mean_range_m:6.1f}")
        time.sleep(0.1)


if __name__ == "__main__":
    main()
