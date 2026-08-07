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
    lidar_name: str = "LidarStructure"
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
                rr = []
                for i in range(0, n * 3, 3):
                    d = math.sqrt(pts[i] ** 2 + pts[i + 1] ** 2 + pts[i + 2] ** 2)
                    if d > 0.5:
                        rr.append(d)
                s.lidar_points = len(rr)
                if rr:
                    s.lidar_mean_range_m = sum(rr) / len(rr)
                    s.lidar_min_range_m = min(rr)
            except Exception as e:
                self._lidar_ok = False
                print(f"[AVISO] LiDAR '{self.lidar_name}' indisponivel ({e}); seguindo sem ele.")
        return s


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
