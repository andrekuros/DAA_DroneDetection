"""
experiment_controller.py — Controlador de Experimentos Cosys-AirSim
===================================================================
Orquestra cenários de aproximação de drone intruso usando a API do
Cosys-AirSim (simGetImages, moveToPositionAsync, simEnableWeather, etc.).

Dependências:
    pip install cosysairsim opencv-python numpy

Uso:
    # Listar cenários sem conectar ao simulador
    python tools/experiment_controller.py --dry-run

    # Executar cenário único
    python tools/experiment_controller.py --scenario frontal_clear_day

    # Executar todos os cenários em sequência
    python tools/experiment_controller.py --all --output-dir dataset/

    # Apenas imagens RGB (sem depth/segmentation)
    python tools/experiment_controller.py --all --image-types scene

Coordenadas NED (AirSim): X=Norte, Y=Leste, Z=para baixo (Z negativo = subindo)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ─── Importação Cosys-AirSim (permite --dry-run sem simulador) ───────────────
try:
    import cosysairsim as airsim
    HAS_AIRSIM = True
    # Cosys-AirSim usa nomes diferentes para os helpers de quaternion
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
        print("[AVISO] cosysairsim não instalado. Apenas --dry-run disponível.")
        print("        Instale com: pip install cosysairsim")

from scenarios import (
    ApproachScenario,
    WeatherConfig,
    get_all_scenarios,
    get_scenario,
)


# ─────────────────────────────────────────────────────────────────────────────
# Configuração global de imagem
# ─────────────────────────────────────────────────────────────────────────────

IMAGE_TYPE_MAP = {
    "scene":   (0, "rgb",   "scene"),    # (airsim ImageType int, subdir, label)
    "depth":   (1, "depth", "depth"),
    "seg":     (5, "seg",   "seg"),
    "ir":      (7, "ir",    "ir"),
    "optflow": (9, "optflow", "optflow"),
}

DEFAULT_IMAGE_TYPES = ["scene", "depth", "seg"]

# Para tipos de ponto flutuante (depth), salva como .pfm em vez de .png
FLOAT_IMAGE_TYPES = {1, 2, 3, 4}   # DepthPlanar, DepthPerspective, DepthVis, Disparity


# ─────────────────────────────────────────────────────────────────────────────
# Configuração do controlador
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ControllerConfig:
    observer_pos: tuple         = (0.0, 0.0, -51.0)  # NED: posicao do observador (altitude 50m para evitar terreno)
    stop_distance_m: float      = 1.0                 # para ao chegar a X m do observador
    capture_interval_s: float   = 0.1                 # intervalo entre capturas (~ 10 fps)
    observer_vehicle_name: str  = "Observer"           # veiculo com a camera (quem "vê" o intruso)
    vehicle_name: str           = "Drone1"            # veiculo intruso (quem se aproxima)
    camera_name: str            = "front_center"      # camera de captura (no Observer)
    image_types: list[str]      = None                # subset de IMAGE_TYPE_MAP.keys()
    output_dir: Path            = Path("dataset")
    airsim_ip: str              = "127.0.0.1"
    takeoff_altitude_m: float   = 50.0                 # altitude de decolagem (metros acima do solo)
    max_frames: int | None       = None                # limite de frames (None = sem limite; util para teste rapido)
    clock_speed: float          = 0.0                  # velocidade do relogio celestial (0 = sol parado)
    enable_lidar_vlp16: bool    = False                # captura Lidar VLP-16 (16ch, 100m)
    enable_lidar_os128: bool    = False                # captura Lidar OS1-128 (128ch, 120m)
    enable_radar: bool          = False                # captura Radar CSM (Echo sensor, 200m)

    def __post_init__(self):
        if self.image_types is None:
            self.image_types = list(DEFAULT_IMAGE_TYPES)


# ─────────────────────────────────────────────────────────────────────────────
# Utilitários
# ─────────────────────────────────────────────────────────────────────────────

def ned_distance(a: tuple, b: tuple) -> float:
    """Distância euclidiana 3D entre dois pontos NED."""
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def save_image(response, out_path: Path, is_float: bool = False):
    """Salva resposta de imagem do AirSim como PNG ou PFM."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if is_float:
        arr = airsim.get_pfm_array(response)
        airsim.write_pfm(str(out_path), arr)
    else:
        # PNG comprimido
        airsim.write_file(str(out_path), response.image_data_uint8)


def write_pfm_manual(path: Path, data: np.ndarray):
    """Fallback para salvar float32 array como PFM sem depender de airsim.write_pfm."""
    h, w = data.shape
    with open(path, "wb") as f:
        f.write(b"Pf\n")
        f.write(f"{w} {h}\n".encode())
        f.write(b"-1.0\n")           # little-endian
        f.write(np.flipud(data).astype(np.float32).tobytes())


# ─────────────────────────────────────────────────────────────────────────────
# Controlador principal
# ─────────────────────────────────────────────────────────────────────────────

class ExperimentController:
    """
    Orquestra experimentos de aproximação de drone no Cosys-AirSim.

    Fluxo por cenário:
        1. Aplica clima e horário
        2. Teleporta o drone intruso para a posição inicial
        3. Decola (se necessário)
        4. Inicia moveToPositionAsync em direção ao observador
        5. Loop de captura de imagens + telemetria até chegar perto
        6. Reset para próximo cenário
    """

    def __init__(self, cfg: ControllerConfig):
        self.cfg = cfg
        self.client: "airsim.MultirotorClient | None" = None

    # ── Leitura de settings.json ──────────────────────────────────────────────

    def _read_vehicle_spawns(self) -> dict[str, tuple]:
        """Le Documents/AirSim/settings.json e retorna {vehicle_name: (X, Y, Z)}
        com a posicao de spawn de cada veiculo. Estas sao os offsets que convertem
        coordenadas locais da API para coordenadas globais NED."""
        home = Path(os.path.expanduser("~"))
        candidates = [
            home / "Documents" / "AirSim" / "settings.json",
            home / "OneDrive" / "Documents" / "AirSim" / "settings.json",
            home / "OneDrive - Personal" / "Documents" / "AirSim" / "settings.json",
        ]
        offsets = {}
        for settings_path in candidates:
            if not settings_path.exists():
                continue
            try:
                with open(settings_path, "r", encoding="utf-8") as f:
                    settings = json.load(f)
                vehicles = settings.get("Vehicles", {})
                for v_name, v_cfg in vehicles.items():
                    offsets[v_name] = (
                        float(v_cfg.get("X", 0.0)),
                        float(v_cfg.get("Y", 0.0)),
                        float(v_cfg.get("Z", 0.0)),
                    )
                print(f"  Settings: {settings_path}")
                return offsets
            except Exception as e:
                print(f"[AVISO] Nao foi possivel ler {settings_path}: {e}")
        print(f"[AVISO] settings.json nao encontrado. Offsets serao (0,0,0).")
        return offsets

    # ── Conexão ───────────────────────────────────────────────────────────────

    def connect(self):
        """
        Conecta ao Cosys-AirSim, detecta veículos disponíveis e habilita controle API.
        Já deixa Observer e Intruso pairando aproximadamente na altitude desejada.
        """
        self.client = airsim.MultirotorClient(ip=self.cfg.airsim_ip)
        self.client.confirmConnection()

        # ── Detecta veículos disponíveis ──────────────────────────────────────
        try:
            available = self.client.listVehicles()
        except Exception:
            available = []

        configured = self.cfg.vehicle_name

        if configured in available:
            vehicle = configured
        else:
            if available:
                vehicle = available[0]
                print(f"\n[AVISO] Intruso '{configured}' nao encontrado. Usando '{vehicle}'.")
                print(f"        Copie config\\cosys_airsim_settings.json para Documents\\AirSim\\ e reinicie.\n")
                self.cfg.vehicle_name = vehicle
            else:
                vehicle = ""
                self.cfg.vehicle_name = vehicle

        obs = self.cfg.observer_vehicle_name
        if obs and available and obs not in available:
            print(f"\n[AVISO] Observer '{obs}' nao encontrado. Captura do intruso '{vehicle}'.")
            self.cfg.observer_vehicle_name = vehicle
        if not obs:
            self.cfg.observer_vehicle_name = vehicle

        # Le settings.json para obter spawn offsets de cada veiculo
        # (simGetVehiclePose retorna coords LOCAIS em Cosys-AirSim, nao globais)
        self._home_offsets = self._read_vehicle_spawns()
        for v_name, off in self._home_offsets.items():
            print(f"  Spawn '{v_name}' (settings.json): X={off[0]:.1f} Y={off[1]:.1f} Z={off[2]:.1f}")

        # Habilita API e coloca os veiculos em voo estavel
        target_z = -self.cfg.takeoff_altitude_m  # NED local: negativo = acima do solo
        seen = set()
        for v_name in [self.cfg.observer_vehicle_name, self.cfg.vehicle_name]:
            if not v_name or v_name in seen:
                continue
            seen.add(v_name)
            try:
                self.client.enableApiControl(True, vehicle_name=v_name)
                self.client.armDisarm(True, vehicle_name=v_name)
                self.client.takeoffAsync(vehicle_name=v_name).join()
                self.client.moveToZAsync(
                    target_z, velocity=5.0, vehicle_name=v_name,
                ).join()
                gpos = self._get_global_position(v_name)
                print(f"  '{v_name}' pronto: global=({gpos[0]:.1f}, {gpos[1]:.1f}, {gpos[2]:.1f})")
            except Exception as e:
                print(f"[AVISO] Nao foi possivel inicializar veiculo '{v_name}': {e}")

        print(
            f"[OK] Cosys-AirSim {self.cfg.airsim_ip} | "
            f"Observer (camera): '{self.cfg.observer_vehicle_name}' | "
            f"Intruso: '{self.cfg.vehicle_name}'"
        )

    # ── Ambiente (clima + horário + vento) ────────────────────────────────────

    # Rastreia APIs não suportadas para não repetir aviso em cada cenário
    _unsupported_apis: set = None

    def _try_call(self, label: str, fn, *args, **kwargs) -> bool:
        """
        Tenta chamar fn(*args, **kwargs). Se falhar, registra aviso uma única vez
        e retorna False. Retorna True em caso de sucesso.
        """
        if self._unsupported_apis is None:
            self.__class__._unsupported_apis = set()
        if label in self._unsupported_apis:
            return False   # já sabemos que não funciona — pula silenciosamente
        try:
            fn(*args, **kwargs)
            return True
        except Exception as e:
            self._unsupported_apis.add(label)
            # Extrai mensagem de erro útil
            msg = str(e).split(".")[-1].strip() if "." in str(e) else str(e)[:80]
            print(f"  [AVISO] API '{label}' não suportada neste ambiente: {msg}")
            return False

    def _apply_environment(self, scenario: ApproachScenario):
        """
        Aplica condições climáticas, vento e horário do dia.
        Cada chamada de API é isolada — falhas não interrompem o cenário.

        Nota: `simSetWind` e APIs de clima requerem que o mapa Unreal tenha
        os atores correspondentes (BP_Sky_Sphere, WeatherActor, etc.).
        O ambiente Blocks (MSBuild2018) suporta todas essas APIs.
        """
        c = self.client

        # ── Horário do dia ────────────────────────────────────────────────────
        self._try_call(
            "simSetTimeOfDay",
            c.simSetTimeOfDay,
            is_enabled=True,
            start_datetime=scenario.time_of_day,
            is_start_datetime_dst=False,
            celestial_clock_speed=self.cfg.clock_speed,   # 0 = congela; >0 acelera sol
            update_interval_secs=60,
            move_sun=True,
        )

        # ── Vento NED ─────────────────────────────────────────────────────────
        wind = airsim.Vector3r(*scenario.wind_ned)
        self._try_call("simSetWind", c.simSetWind, wind)

        # ── Clima (sempre reseta para garantir que cenario anterior nao persiste)
        w = scenario.weather
        if self._try_call("simEnableWeather", c.simEnableWeather, True):
            self._try_call("rain",  c.simSetWeatherParameter,
                           airsim.WeatherParameter.Rain,  w.rain)
            self._try_call("fog",   c.simSetWeatherParameter,
                           airsim.WeatherParameter.Fog,   w.fog)
            self._try_call("dust",  c.simSetWeatherParameter,
                           airsim.WeatherParameter.Dust,  w.dust)
            self._try_call("snow",  c.simSetWeatherParameter,
                           airsim.WeatherParameter.Snow,  w.snow)

    # ── Posicionamento ────────────────────────────────────────────────────────

    def _position_observer_at_fixed_pose(self):
        """
        Coloca o veiculo Observer (camera) na posicao fixa do observador e
        garante que ele fique pairando aproximadamente nessa altitude.
        """
        if not self.cfg.observer_vehicle_name or self.cfg.observer_vehicle_name == self.cfg.vehicle_name:
            return

        ox, oy, oz = self.cfg.observer_pos
        pose = airsim.Pose(
            position_val=airsim.Vector3r(ox, oy, oz),
            orientation_val=airsim.to_quaternion(0, 0, 0),
        )

        # Teleporta para a posicao desejada
        self._try_call(
            "simSetVehiclePose(Observer)",
            self.client.simSetVehiclePose,
            pose,
            ignore_collision=True,
            vehicle_name=self.cfg.observer_vehicle_name,
        )

        # Liga motores e manda ir para a mesma altitude (evita cair no chao)
        try:
            self.client.armDisarm(True, vehicle_name=self.cfg.observer_vehicle_name)
            self.client.moveToZAsync(
                oz,
                velocity=2.0,
                vehicle_name=self.cfg.observer_vehicle_name,
            ).join()
        except Exception as e:
            print(f"[AVISO] Nao foi possivel estabilizar Observer: {e}")

        time.sleep(0.3)

    def _position_drone_for_scenario(self, target_pos: tuple):
        """
        Posiciona o drone para o início do cenário.
        Tenta usar teleport (simSetVehiclePose). Se falhar (bad cast),
        usa voo autônomo (moveToPositionAsync) até o ponto inicial.
        """
        x, y, z = target_pos
        pose = airsim.Pose(
            position_val=airsim.Vector3r(x, y, z),
            orientation_val=airsim.to_quaternion(0, 0, 0)
        )

        print(f"  -> Posicionando drone em {target_pos}...")

        # Tenta Teleport (instantâneo, mas sensível a versão)
        teleport_ok = self._try_call(
            "simSetVehiclePose",
            self.client.simSetVehiclePose,
            pose,
            ignore_collision=True,
            vehicle_name=self.cfg.vehicle_name
        )

        if not teleport_ok:
            # Fallback: Voo para a posição inicial
            print("  -> Usando fallback: voando ate o ponto de partida...")
            # Garante que está armado e decolado
            self.client.armDisarm(True, vehicle_name=self.cfg.vehicle_name)
            self.client.takeoffAsync(vehicle_name=self.cfg.vehicle_name).join()

            # Voa rápido até a posição inicial do cenário
            self.client.moveToPositionAsync(
                x, y, z,
                velocity=20.0,
                vehicle_name=self.cfg.vehicle_name
            ).join()

        # Pequena pausa para estabilizar
        time.sleep(1.0)

    # ── Decolagem ─────────────────────────────────────────────────────────────

    def _takeoff(self):
        c = self.client
        c.armDisarm(True, vehicle_name=self.cfg.vehicle_name)
        c.takeoffAsync(timeout_sec=10, vehicle_name=self.cfg.vehicle_name).join()
        time.sleep(0.5)

    # ── Captura de imagens ────────────────────────────────────────────────────

    def _build_image_requests(self) -> list:
        """Monta lista de ImageRequest para os tipos configurados."""
        requests = []
        for key in self.cfg.image_types:
            if key not in IMAGE_TYPE_MAP:
                print(f"[AVISO] Tipo de imagem desconhecido: '{key}' — ignorado")
                continue
            img_type_int, _, _ = IMAGE_TYPE_MAP[key]
            is_float = img_type_int in FLOAT_IMAGE_TYPES
            requests.append(
                airsim.ImageRequest(
                    self.cfg.camera_name,
                    img_type_int,
                    pixels_as_float=is_float,
                    compress=not is_float,
                )
            )
        return requests

    def _save_frame(self, responses, frame_idx: int, run_dir: Path):
        """Salva todos os tipos de imagem de um frame."""
        for resp, key in zip(responses, self.cfg.image_types):
            img_type_int, subdir, _ = IMAGE_TYPE_MAP[key]
            is_float = img_type_int in FLOAT_IMAGE_TYPES
            ext = ".pfm" if is_float else ".png"
            out_path = run_dir / subdir / f"frame_{frame_idx:06d}{ext}"
            save_image(resp, out_path, is_float=is_float)

    _LIDAR_SENSORS = [
        ("enable_lidar_vlp16", "LidarVLP16", "lidar_vlp16"),
        ("enable_lidar_os128", "LidarOS128", "lidar_os128"),
    ]

    def _save_lidar_frame(self, frame_idx: int, run_dir: Path):
        """Captura e salva point clouds dos Lidars habilitados no Observer."""
        for cfg_flag, sensor_name, subfolder in self._LIDAR_SENSORS:
            if not getattr(self.cfg, cfg_flag, False):
                continue
            try:
                lidar_data = self.client.getLidarData(
                    sensor_name,
                    vehicle_name=self.cfg.observer_vehicle_name,
                )
                if len(lidar_data.point_cloud) < 3:
                    continue
                points = np.array(lidar_data.point_cloud, dtype=np.float32)
                points = points.reshape(-1, 3)
                out_dir = run_dir / subfolder
                out_dir.mkdir(parents=True, exist_ok=True)
                np.save(out_dir / f"frame_{frame_idx:06d}.npy", points)
            except Exception as e:
                if frame_idx == 0:
                    print(f"  [AVISO] Lidar {sensor_name}: {e}")

    def _save_radar_frame(self, frame_idx: int, run_dir: Path):
        """Captura e salva point cloud do Radar (Echo sensor) do Observer."""
        if not self.cfg.enable_radar:
            return
        try:
            echo_data = self.client.getEchoData(
                "RadarCSM",
                vehicle_name=self.cfg.observer_vehicle_name,
            )
            if len(echo_data.point_cloud) < 6:
                return
            points = np.array(echo_data.point_cloud, dtype=np.float32)
            points = points.reshape(-1, 6)  # x,y,z, attenuation, distance, reflections
            out_dir = run_dir / "radar"
            out_dir.mkdir(parents=True, exist_ok=True)
            np.save(out_dir / f"frame_{frame_idx:06d}.npy", points)
        except Exception as e:
            if frame_idx == 0:
                print(f"  [AVISO] Radar: {e}")

    # ── Telemetria ────────────────────────────────────────────────────────────

    def _get_telemetry(self) -> dict:
        state = self.client.getMultirotorState(vehicle_name=self.cfg.vehicle_name)
        kin   = state.kinematics_estimated

        pos = kin.position
        vel = kin.linear_velocity
        ang = kin.angular_velocity
        ori = kin.orientation  # quaternion

        # Converte quaternion → pitch/roll/yaw
        pitch, roll, yaw = airsim.to_eularian_angles(ori)

        return {
            "timestamp_s": time.time(),
            "x_m":   pos.x_val,
            "y_m":   pos.y_val,
            "z_m":   pos.z_val,
            "vx_ms": vel.x_val,
            "vy_ms": vel.y_val,
            "vz_ms": vel.z_val,
            "roll_rad":  roll,
            "pitch_rad": pitch,
            "yaw_rad":   yaw,
            "wx": ang.x_val,
            "wy": ang.y_val,
            "wz": ang.z_val,
        }

    # ── Execução de cenário ───────────────────────────────────────────────────

    def _get_local_position(self, vehicle_name: str) -> tuple:
        """Retorna (x, y, z) em coordenadas LOCAIS do veiculo (frame do API)."""
        p = self.client.simGetVehiclePose(vehicle_name=vehicle_name).position
        return (p.x_val, p.y_val, p.z_val)

    def _get_global_position(self, vehicle_name: str) -> tuple:
        """Retorna (x, y, z) em coordenadas GLOBAIS NED.
        XY global = XY local + spawn XY offset (do settings.json).
        Z: a API ja retorna a altitude local relativa ao home — usamos direto
        pois ambos os drones usam o mesmo moveToZAsync target."""
        lx, ly, lz = self._get_local_position(vehicle_name)
        hx, hy, _ = self._home_offsets.get(vehicle_name, (0.0, 0.0, 0.0))
        return (lx + hx, ly + hy, lz)

    def _to_local(self, vehicle_name: str, global_xyz: tuple) -> tuple:
        """Converte coordenadas globais NED para o frame local do veiculo
        (necessario para moveToPositionAsync). Apenas XY tem offset."""
        hx, hy, _ = self._home_offsets.get(vehicle_name, (0.0, 0.0, 0.0))
        return (global_xyz[0] - hx, global_xyz[1] - hy, global_xyz[2])

    def run_scenario(self, scenario: ApproachScenario, run_idx: int):
        """Executa um unico cenario de aproximacao e salva imagens + telemetria."""
        print(f"\n{'='*70}")
        print(f"  CENARIO [{run_idx:03d}]: {scenario.name}")
        print(f"  {scenario.description}")
        print(f"  {scenario.summary()}")
        print(f"{'='*70}")

        cfg = self.cfg
        run_dir = cfg.output_dir / f"run_{run_idx:03d}_{scenario.name}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # 1. Ambiente
        self._apply_environment(scenario)

        # 2. Ler posicao REAL do Observer no simulador (nao usar cfg.observer_pos)
        obs_real = self._get_global_position(cfg.observer_vehicle_name)
        print(f"  -> Observer posicao atual (global): ({obs_real[0]:.1f}, {obs_real[1]:.1f}, {obs_real[2]:.1f})")

        # 3. Calcular posicao alvo do intruso relativa ao Observer REAL
        start_pos_global = scenario.start_position_ned(observer_ned=obs_real)
        target_z_global = start_pos_global[2]

        # Intruso voa para a posicao de partida do cenario
        intr_local = self._to_local(cfg.vehicle_name, start_pos_global)
        print(f"  -> Intruso:  global={tuple(round(v,1) for v in start_pos_global)} "
              f"local={tuple(round(v,1) for v in intr_local)}")

        # 4. Voar intruso para a posicao inicial (Observer ja esta onde deve)
        f_intr = self.client.moveToPositionAsync(
            intr_local[0], intr_local[1], intr_local[2], velocity=20.0,
            timeout_sec=60,
            vehicle_name=cfg.vehicle_name,
        )
        f_intr.join()

        # 5. Confirmar posicoes reais (GLOBAIS) antes de comecar
        obs_real = self._get_global_position(cfg.observer_vehicle_name)
        intr_real = self._get_global_position(cfg.vehicle_name)
        initial_dist = math.hypot(intr_real[0] - obs_real[0], intr_real[1] - obs_real[1])
        print(f"  -> Observer GLOBAL: ({obs_real[0]:.1f}, {obs_real[1]:.1f}, {obs_real[2]:.1f})")
        print(f"  -> Intruso  GLOBAL: ({intr_real[0]:.1f}, {intr_real[1]:.1f}, {intr_real[2]:.1f})")
        print(f"  -> Distancia inicial real: {initial_dist:.1f}m")

        time.sleep(0.5)

        # 6. Intruso voa ALEM do observador (para passar por ele, nao parar antes)
        #    front_dir: direcao que o observer olha (de obs -> intruso start)
        approach_dx = intr_real[0] - obs_real[0]
        approach_dy = intr_real[1] - obs_real[1]
        approach_len = math.hypot(approach_dx, approach_dy)
        if approach_len > 0.1:
            nx, ny = approach_dx / approach_len, approach_dy / approach_len
        else:
            nx, ny = 1.0, 0.0
        overshoot = 200.0
        fly_beyond_global = (
            obs_real[0] - nx * overshoot,
            obs_real[1] - ny * overshoot,
            target_z_global,
        )
        intr_fly_target = self._to_local(cfg.vehicle_name, fly_beyond_global)
        self.client.moveToPositionAsync(
            intr_fly_target[0], intr_fly_target[1], intr_fly_target[2],
            velocity=scenario.speed_ms,
            timeout_sec=600,
            drivetrain=airsim.DrivetrainType.ForwardOnly,
            yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0),
            vehicle_name=cfg.vehicle_name,
        )

        # Observer mantem voo estavel (paira na posicao atual, em LOCAL)
        if cfg.observer_vehicle_name and cfg.observer_vehicle_name != cfg.vehicle_name:
            obs_hover = self._to_local(cfg.observer_vehicle_name, obs_real)
            self.client.moveToPositionAsync(
                obs_hover[0], obs_hover[1], obs_hover[2],
                velocity=1.0,
                timeout_sec=600,
                vehicle_name=cfg.observer_vehicle_name,
            )

        # 7. Loop de captura (NAO depende de future._set_flag)
        requests = self._build_image_requests()
        telemetry_rows = []
        frame_idx = 0
        last_capture = time.perf_counter()

        print(f"  -> Capturando {len(self.cfg.image_types)} tipo(s) de imagem...")

        while True:
            now = time.perf_counter()
            if now - last_capture < cfg.capture_interval_s:
                time.sleep(0.005)
                continue
            last_capture = now

            # Imagens da camera do Observer
            try:
                responses = self.client.simGetImages(
                    requests, vehicle_name=cfg.observer_vehicle_name
                )
                self._save_frame(responses, frame_idx, run_dir)
            except Exception as e:
                print(f"  [AVISO] Erro na captura frame {frame_idx}: {e}")

            # Lidar point cloud
            self._save_lidar_frame(frame_idx, run_dir)

            # Radar (Echo sensor)
            self._save_radar_frame(frame_idx, run_dir)

            # Telemetria do intruso (local frame, para registro)
            tele = self._get_telemetry()
            tele["frame"] = frame_idx
            tele["scenario"] = scenario.name

            # Posicoes GLOBAIS de ambos os drones
            try:
                intr_gx, intr_gy, intr_gz = self._get_global_position(cfg.vehicle_name)
            except Exception:
                intr_gx, intr_gy, intr_gz = intr_real
            try:
                obs_gx, obs_gy, obs_gz = self._get_global_position(cfg.observer_vehicle_name)
            except Exception:
                obs_gx, obs_gy, obs_gz = obs_real

            # Distancia real entre os dois drones (plano XY, coords globais)
            dist = math.hypot(intr_gx - obs_gx, intr_gy - obs_gy)

            # Posicao relativa do intruso ao observador, projetada na direcao de aproximacao
            # Positivo = intruso ainda esta a frente; Negativo = passou por tras
            rel_dot = (intr_gx - obs_gx) * nx + (intr_gy - obs_gy) * ny

            tele["intr_global_x"] = intr_gx
            tele["intr_global_y"] = intr_gy
            tele["intr_global_z"] = intr_gz
            tele["obs_global_x"] = obs_gx
            tele["obs_global_y"] = obs_gy
            tele["obs_global_z"] = obs_gz
            tele["dist_xy_m"] = dist
            tele["behind_dot"] = rel_dot
            telemetry_rows.append(tele)

            if frame_idx % 10 == 0:
                status = "FRENTE" if rel_dot > 0 else "ATRAS"
                print(
                    f"    frame {frame_idx:05d} | "
                    f"dist={dist:.1f}m | "
                    f"{status} ({rel_dot:+.1f}) | "
                    f"intr=({intr_gx:.1f},{intr_gy:.1f},{intr_gz:.1f}) | "
                    f"obs=({obs_gx:.1f},{obs_gy:.1f},{obs_gz:.1f})"
                )

            frame_idx += 1

            # Criterio de parada: intruso passou por tras do observador
            if rel_dot < -cfg.stop_distance_m:
                print(f"  -> Intruso passou {abs(rel_dot):.1f}m atras do observador. Proximo cenario.")
                break
            # Limite de frames
            if cfg.max_frames is not None and frame_idx >= cfg.max_frames:
                print(f"  -> Limite de {cfg.max_frames} frames. Encerrando cenario.")
                break

        # Cancela movimento de ambos
        try:
            self.client.cancelLastTask(vehicle_name=cfg.vehicle_name)
        except Exception:
            pass
        if cfg.observer_vehicle_name and cfg.observer_vehicle_name != cfg.vehicle_name:
            try:
                self.client.cancelLastTask(vehicle_name=cfg.observer_vehicle_name)
            except Exception:
                pass

        # 6. Salva telemetria CSV
        if telemetry_rows:
            csv_path = run_dir / "telemetry.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(telemetry_rows[0].keys()))
                writer.writeheader()
                writer.writerows(telemetry_rows)
            print(f"  -> {frame_idx} frames | telemetria: {csv_path}")
        else:
            print("  [AVISO] Nenhum frame capturado.")

        # Retornar drones para posicoes de spawn (settings.json)
        target_z = -cfg.takeoff_altitude_m
        print("  -> Retornando drones para posicoes iniciais...")
        futures = []
        for v_name in set(filter(None, [cfg.vehicle_name, cfg.observer_vehicle_name])):
            spawn = self._home_offsets.get(v_name, (0.0, 0.0, 0.0))
            local_target = self._to_local(v_name, (spawn[0], spawn[1], target_z))
            try:
                f = self.client.moveToPositionAsync(
                    local_target[0], local_target[1], local_target[2],
                    velocity=20.0, timeout_sec=60, vehicle_name=v_name,
                )
                futures.append(f)
            except Exception:
                pass
        for f in futures:
            try:
                f.join()
            except Exception:
                pass

    # ── Execução batch ────────────────────────────────────────────────────────

    def run_all(self, scenarios: list[ApproachScenario]):
        """Executa todos os cenários em sequência."""
        total = len(scenarios)
        print(f"\n[INFO] Iniciando {total} cenario(s)...")
        for i, scenario in enumerate(scenarios, 1):
            try:
                self.run_scenario(scenario, run_idx=i)
            except KeyboardInterrupt:
                print("\n[INTERROMPIDO] Encerrando experimento.")
                break
            except Exception as e:
                print(f"[ERRO] Cenario '{scenario.name}': {e}")
                # Tenta recuperar sem reset (que mata a conexao)
                for v_name in set(filter(None, [self.cfg.vehicle_name, self.cfg.observer_vehicle_name])):
                    try:
                        self.client.enableApiControl(True, vehicle_name=v_name)
                        self.client.hoverAsync(vehicle_name=v_name).join()
                    except Exception:
                        pass
        print(f"\n[CONCLUIDO] {total} cenario(s) | Dataset: {self.cfg.output_dir.resolve()}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Controlador de experimentos Cosys-AirSim – aproximação de drone intruso."
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Lista cenários e posições sem conectar ao simulador.",
    )
    p.add_argument(
        "--scenario", "-s", type=str, default=None,
        help="Nome de um cenario a executar (use --list para ver opcoes).",
    )
    p.add_argument(
        "--scenarios", nargs="+", type=str, default=None, metavar="NAME",
        help="Lista de cenarios a executar em sequencia (ex: frontal_clear_day frontal_fog).",
    )
    p.add_argument(
        "--all", "-a", action="store_true",
        help="Executa todos os cenarios em sequencia.",
    )
    p.add_argument(
        "--list", "-l", action="store_true",
        help="Lista todos os cenários disponíveis.",
    )
    p.add_argument(
        "--output-dir", "-o", type=Path, default=Path("dataset"),
        help="Diretório de saída do dataset (padrão: dataset/).",
    )
    p.add_argument(
        "--observer-pos", nargs=3, type=float, metavar=("X", "Y", "Z"),
        default=[0.0, 0.0, -50.0],
        help="Posicao NED do observador em metros (padrao: 0 0 -50 = 50m altura).",
    )
    p.add_argument(
        "--observer-vehicle", type=str, default="Observer",
        help="Veiculo com a camera que ve o intruso (padrao: Observer).",
    )
    p.add_argument(
        "--stop-dist", type=float, default=3.0,
        help="Distância (m) para interromper aproximação (padrão: 5).",
    )
    p.add_argument(
        "--capture-interval", type=float, default=0.1,
        help="Intervalo em segundos entre capturas de imagem (padrão: 0.1 = 10 fps).",
    )
    p.add_argument(
        "--image-types", nargs="+",
        choices=list(IMAGE_TYPE_MAP.keys()),
        default=DEFAULT_IMAGE_TYPES,
        help=f"Tipos de imagem a capturar (padrão: {DEFAULT_IMAGE_TYPES}). "
             "Opções: scene depth seg ir optflow.",
    )
    p.add_argument(
        "--lidar-vlp16", action="store_true",
        help="Captura point cloud do Lidar VLP-16 (16ch, 100m) no Observer.",
    )
    p.add_argument(
        "--lidar-os128", action="store_true",
        help="Captura point cloud do Lidar OS1-128 (128ch, 120m) no Observer.",
    )
    p.add_argument(
        "--radar", action="store_true",
        help="Captura point cloud do Radar CSM (Echo sensor, 200m) no Observer.",
    )
    p.add_argument(
        "--vehicle", type=str, default="Drone1",
        help="Nome do veículo intruso no settings.json do Cosys-AirSim (padrão: Drone1).",
    )
    p.add_argument(
        "--camera", type=str, default="front_center",
        help="Nome da câmera de captura (padrão: front_center).",
    )
    p.add_argument(
        "--ip", type=str, default="127.0.0.1",
        help="IP do host Cosys-AirSim (padrão: 127.0.0.1).",
    )
    p.add_argument(
        "--max-frames", type=int, default=None, metavar="N",
        help="Limite de frames por cenario (teste rapido; ex: 50).",
    )
    p.add_argument(
        "--clock-speed", type=float, default=0.0,
        help="Velocidade do relogio celestial (0 = sol parado, 1 = tempo real, >1 acelera apenas iluminacao).",
    )
    p.add_argument(
        "--list-vehicles", action="store_true",
        help="Lista veículos disponíveis no Cosys-AirSim e encerra.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    all_scenarios = get_all_scenarios()

    # ── Listar veículos no simulador ─────────────────────────────────────────
    if args.list_vehicles:
        if not HAS_AIRSIM:
            print("[ERRO] Instale cosysairsim para listar veículos.")
            sys.exit(1)
        tmp = airsim.MultirotorClient(ip=args.ip)
        tmp.confirmConnection()
        try:
            vehicles = tmp.listVehicles()
            print(f"\nVeículos disponíveis no Cosys-AirSim ({args.ip}):")
            for v in vehicles:
                print(f"  - {v}")
        except Exception as e:
            print(f"[ERRO] Não foi possível listar veículos: {e}")
        return

    # ── Listar cenários ───────────────────────────────────────────────────────
    if args.list or args.dry_run:
        observer = tuple(args.observer_pos)
        print(f"\n{'#':>3}  {'Nome':<30} {'Pos Inicial NED (x,y,z)':^30}  Resumo")
        print("─" * 110)
        for i, s in enumerate(all_scenarios, 1):
            pos = s.start_position_ned(observer)
            pos_str = f"({pos[0]:+6.1f}, {pos[1]:+6.1f}, {pos[2]:+6.1f})"
            print(f"{i:>3}  {s.name:<30} {pos_str:^30}  {s.summary()}")
        print(f"\nTotal: {len(all_scenarios)} cenários | Observador NED: {observer}")
        if args.dry_run:
            return

    # ── Seleciona cenários a executar ─────────────────────────────────────────
    if args.scenarios:
        scenarios_to_run = []
        for name in args.scenarios:
            try:
                scenarios_to_run.append(get_scenario(name))
            except KeyError as e:
                print(e)
                sys.exit(1)
    elif args.scenario:
        try:
            scenarios_to_run = [get_scenario(args.scenario)]
        except KeyError as e:
            print(e)
            sys.exit(1)
    elif args.all:
        scenarios_to_run = all_scenarios
    elif not args.list:
        print("[ERRO] Especifique --scenario <nome>, --scenarios <n1 n2 ...>, --all, --list ou --dry-run.")
        sys.exit(1)
    else:
        return

    # ── Valida AirSim ────────────────────────────────────────────────────────
    if not HAS_AIRSIM:
        print("[ERRO] Instale cosysairsim: pip install cosysairsim")
        sys.exit(1)

    # ── Monta configuração e executa ──────────────────────────────────────────
    cfg = ControllerConfig(
        observer_pos=tuple(args.observer_pos),
        stop_distance_m=args.stop_dist,
        capture_interval_s=args.capture_interval,
        observer_vehicle_name=args.observer_vehicle,
        vehicle_name=args.vehicle,
        camera_name=args.camera,
        image_types=args.image_types,
        output_dir=args.output_dir,
        airsim_ip=args.ip,
        max_frames=args.max_frames,
        clock_speed=args.clock_speed,
        enable_lidar_vlp16=args.lidar_vlp16,
        enable_lidar_os128=args.lidar_os128,
        enable_radar=args.radar,
    )

    controller = ExperimentController(cfg)
    controller.connect()

    if len(scenarios_to_run) == 1:
        controller.run_scenario(scenarios_to_run[0], run_idx=1)
    else:
        controller.run_all(scenarios_to_run)


if __name__ == "__main__":
    main()
