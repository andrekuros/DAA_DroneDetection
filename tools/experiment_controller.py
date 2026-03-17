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

        # Habilita API e já coloca os veículos em voo estável
        for v_name, z_target in [
            (self.cfg.observer_vehicle_name, self.cfg.observer_pos[2]),
            (self.cfg.vehicle_name, self.cfg.observer_pos[2]),
        ]:
            if not v_name:
                continue
            try:
                self.client.enableApiControl(True, vehicle_name=v_name)
                self.client.armDisarm(True, vehicle_name=v_name)
                # Decola e sobe/ajusta para a altitude alvo
                self.client.takeoffAsync(vehicle_name=v_name).join()
                self.client.moveToZAsync(
                    z_target,
                    velocity=2.0,
                    vehicle_name=v_name,
                ).join()
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

        # ── Clima ─────────────────────────────────────────────────────────────
        w = scenario.weather
        if not w.is_clear():
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

    def run_scenario(self, scenario: ApproachScenario, run_idx: int):
        """Executa um único cenário de aproximação e salva imagens + telemetria."""
        print(f"\n{'='*70}")
        print(f"  CENÁRIO [{run_idx:03d}]: {scenario.name}")
        print(f"  {scenario.description}")
        print(f"  {scenario.summary()}")
        print(f"{'='*70}")

        cfg = self.cfg
        run_dir = cfg.output_dir / f"run_{run_idx:03d}_{scenario.name}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # 1. Ambiente
        self._apply_environment(scenario)

        # 2. Posiciona o Observer (camera) na posicao fixa; intruso na posicao inicial
        self._position_observer_at_fixed_pose()
        start_pos = scenario.start_position_ned(observer_ned=cfg.observer_pos)
        self._position_drone_for_scenario(start_pos)

        # 3. Estabilização final
        self.client.armDisarm(True, vehicle_name=cfg.vehicle_name)
        # Garante altitude do cenário
        self.client.moveToZAsync(
            start_pos[2], velocity=3.0,
            vehicle_name=cfg.vehicle_name,
        ).join()
        time.sleep(0.5)

        # 4. Comandos de aproximação (assíncronos)
        #    - Intruso vai em direção ao observador, mantendo altitude do cenário
        #    - Observer também é controlado pelo SimpleFlight para não cair
        ox, oy, _ = cfg.observer_pos
        target_z = start_pos[2]  # altitude fixa definida pelo cenário
        future = self.client.moveToPositionAsync(
            ox, oy, target_z,
            velocity=scenario.speed_ms,
            timeout_sec=300,
            drivetrain=airsim.DrivetrainType.ForwardOnly,
            yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0),
            vehicle_name=cfg.vehicle_name,
        )

        # Mantém o Observer em voo controlado na mesma altitude
        if cfg.observer_vehicle_name and cfg.observer_vehicle_name != cfg.vehicle_name:
            try:
                self.client.armDisarm(True, vehicle_name=cfg.observer_vehicle_name)
                self.client.moveToPositionAsync(
                    ox, oy, target_z,
                    velocity=3.0,
                    timeout_sec=300,
                    drivetrain=airsim.DrivetrainType.ForwardOnly,
                    yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0),
                    vehicle_name=cfg.observer_vehicle_name,
                )
            except Exception as e:
                print(f"[AVISO] Nao foi possivel mover Observer com SimpleFlight: {e}")

        # 5. Loop de captura
        requests = self._build_image_requests()
        telemetry_rows = []
        frame_idx = 0
        last_capture = time.perf_counter()

        print(f"  -> Capturando {len(self.cfg.image_types)} tipo(s) de imagem...")

        while not future._set_flag:
            now = time.perf_counter()
            if now - last_capture < cfg.capture_interval_s:
                time.sleep(0.005)
                continue
            last_capture = now

            # Imagens da camera do Observer (quem ve o intruso se aproximando)
            try:
                responses = self.client.simGetImages(
                    requests, vehicle_name=cfg.observer_vehicle_name
                )
                self._save_frame(responses, frame_idx, run_dir)
            except Exception as e:
                print(f"  [AVISO] Erro na captura frame {frame_idx}: {e}")

            # Telemetria do intruso
            tele = self._get_telemetry()
            tele["frame"] = frame_idx
            tele["scenario"] = scenario.name
            telemetry_rows.append(tele)

            # Telemetria real do observer (drone proprio) se existir
            obs_x, obs_y, obs_z = cfg.observer_pos
            if cfg.observer_vehicle_name:
                try:
                    obs_state = self.client.getMultirotorState(
                        vehicle_name=cfg.observer_vehicle_name
                    )
                    obs_pos = obs_state.kinematics_estimated.position
                    obs_x = obs_pos.x_val
                    obs_y = obs_pos.y_val
                    obs_z = obs_pos.z_val
                except Exception:
                    pass

            # Distância intruso–observer no plano horizontal (X–Y)
            dx = tele["x_m"] - obs_x
            dy = tele["y_m"] - obs_y
            dist = math.hypot(dx, dy)

            if frame_idx % 10 == 0:
                print(
                    f"    frame {frame_idx:05d} | "
                    f"dist_xy={dist:.1f}m | "
                    f"z_intruso={tele['z_m']:.1f}m | "
                    f"z_observer={obs_z:.1f}m | "
                    f"vel=({tele['vx_ms']:.1f},{tele['vy_ms']:.1f},{tele['vz_ms']:.1f})"
                )

            frame_idx += 1

            # Para quando chega perto do observador (distancia no plano XY)
            if dist <= cfg.stop_distance_m:
                print(f"  -> Chegou a {dist:.1f}m do observador. Encerrando cenario.")
                break
            # Limite de frames (teste rapido)
            if cfg.max_frames is not None and frame_idx >= cfg.max_frames:
                print(f"  -> Limite de {cfg.max_frames} frames. Encerrando cenario.")
                break

        # Cancela movimento se ainda ativo
        try:
            self.client.cancelLastTask(vehicle_name=cfg.vehicle_name)
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

        # 7. Reset para proximo cenario
        self.client.reset()
        self.client.enableApiControl(True, vehicle_name=cfg.vehicle_name)
        if cfg.observer_vehicle_name and cfg.observer_vehicle_name != cfg.vehicle_name:
            try:
                self.client.enableApiControl(True, vehicle_name=cfg.observer_vehicle_name)
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
                try:
                    self.client.reset()
                    self.client.enableApiControl(True, vehicle_name=self.cfg.vehicle_name)
                    if self.cfg.observer_vehicle_name and self.cfg.observer_vehicle_name != self.cfg.vehicle_name:
                        self.client.enableApiControl(True, vehicle_name=self.cfg.observer_vehicle_name)
                except Exception:
                    pass
        print(f"\n[CONCLUÍDO] Dataset salvo em: {self.cfg.output_dir.resolve()}")


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
        help="Nome do cenário a executar (use --list para ver opções).",
    )
    p.add_argument(
        "--all", "-a", action="store_true",
        help="Executa todos os cenários em sequência.",
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
    if args.scenario:
        try:
            scenarios_to_run = [get_scenario(args.scenario)]
        except KeyError as e:
            print(e)
            sys.exit(1)
    elif args.all:
        scenarios_to_run = all_scenarios
    elif not args.list:
        print("[ERRO] Especifique --scenario <nome>, --all, --list ou --dry-run.")
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
    )

    controller = ExperimentController(cfg)
    controller.connect()

    if len(scenarios_to_run) == 1:
        controller.run_scenario(scenarios_to_run[0], run_idx=1)
    else:
        controller.run_all(scenarios_to_run)


if __name__ == "__main__":
    main()
