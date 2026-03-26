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

    # Teste rápido: primeiros N cenários (ordem em scenarios.py)
    # Mais frames por percurso: --dense-capture (~20 Hz) ou --capture-interval 0.033 (~30 Hz)
    python tools/experiment_controller.py --first-n 3 --output-dir dataset_smoke/ --dense-capture

    # Mesma rota do intruso; vistas por rotação do Observer (azimuth do cenario = yaw).
    # Com --rotate-observer o desvio lateral padrao e 4 m no plano XY; evita-colisao com o Observer e por altitude
    # (--avoid-altitude-ned-m) so se |dz| <= --avoid-same-level-tol-m.
    python tools/experiment_controller.py --first-n 3 --rotate-observer --fixed-intruder-azimuth 0 --vertical-separation 2.5

    # Apenas imagens RGB (sem depth/segmentation)
    python tools/experiment_controller.py --all --image-types scene

Coordenadas NED (AirSim): X=Norte, Y=Leste, Z=para baixo (Z negativo = subindo)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
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
    stop_distance_m: float      = 1.0                 # fim da captura quando rel_dot < -X (intruso X m além do plano do obs, eixo de aproximação)
    capture_interval_s: float   = 0.1                 # intervalo entre capturas (~ 10 fps)
    observer_vehicle_name: str  = "Observer"           # veiculo com a camera (quem "vê" o intruso)
    vehicle_name: str           = "Drone1"            # veiculo intruso (quem se aproxima)
    camera_name: str            = "front_center"      # camera de captura (no Observer)
    image_types: list[str]      = None                # subset de IMAGE_TYPE_MAP.keys()
    output_dir: Path            = Path("dataset")
    airsim_ip: str              = "127.0.0.1"
    takeoff_altitude_m: float   = 50.0                 # altitude de decolagem (metros acima do solo)
    # NED: Z positivo = para baixo. +1.0 => intruso 1 m mais baixo que o plano do cenario (reduz colisao com o observador)
    vertical_separation_ned_m: float = 1.0
    # True: bearing do intruso = scenario.azimuth_deg; yaw Observer = bearing + rotate_observer_offset_deg.
    rotate_observer_for_geometry: bool = False
    fixed_intruder_azimuth_deg: float = 0.0  # usado sem --rotate-observer; com rotate, ignorado
    # Desloca o ponto de partida no plano XY, perpendicular a obs->partida (a rota continua reta; nao passa pelo XY do obs)
    approach_lateral_offset_m: float = 0.0
    # Dois segmentos: 1) ate intruder_avoid_pass_m no eixo; 2) so se coplanares (|dz|<=tol), passa atras com delta Z NED
    intruder_avoid_pass_m: float = 10.0
    intruder_avoid_altitude_ned_m: float = 5.0  # + = intruso mais "baixo" no NED (desce)
    intruder_avoid_same_level_tol_m: float = 3.0  # so aplica evita-altitude se |z_obs - z_intr| <= isto
    max_frames: int | None       = None                # limite de frames (None = sem limite; util para teste rapido)
    clock_speed: float          = 0.0                  # velocidade do relogio celestial (0 = sol parado)
    enable_lidar_vlp16: bool    = False                # captura Lidar VLP-16 (16ch, 100m)
    enable_lidar_os128: bool    = False                # captura Lidar OS1-128 (128ch, 120m)
    enable_radar: bool          = False                # captura Radar CSM (Echo sensor, 200m)
    camera_names: list[str] | None = None              # lista de cameras para batch (None = usa camera_name)
    # FOV (graus) gravado na telemetria se simGetCameraInfo falhar ou for duvidoso; alinhar ao FOV_Degrees do settings.json.
    camera_fov_fallback_deg: float = 120.0

    def __post_init__(self):
        if self.image_types is None:
            self.image_types = list(DEFAULT_IMAGE_TYPES)


# ─────────────────────────────────────────────────────────────────────────────
# Utilitários
# ─────────────────────────────────────────────────────────────────────────────

def ned_distance(a: tuple, b: tuple) -> float:
    """Distância euclidiana 3D entre dois pontos NED."""
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def intruder_start_global_ned(
    observer_ned: tuple[float, float, float],
    distance_m: float,
    azimuth_deg: float,
    altitude_offset_m: float,
) -> tuple[float, float, float]:
    """Posição NED inicial do intruso (mesma geometria que ApproachScenario.start_position_ned)."""
    az_rad = math.radians(azimuth_deg)
    dx = distance_m * math.cos(az_rad)
    dy = distance_m * math.sin(az_rad)
    dz = -altitude_offset_m
    ox, oy, oz = observer_ned
    return (ox + dx, oy + dy, oz + dz)


def diagnose_segmentation_response(resp) -> tuple[bool, str]:
    """
    Retorna (provavelmente_preta, mensagem_curta).
    Seg preta costuma ser cena Unreal sem IDs de segmentacao (nao e bug do PNG em si).
    """
    raw = getattr(resp, "image_data_uint8", None)
    if raw is None:
        return True, "sem image_data_uint8"
    if len(raw) == 0:
        return True, "buffer 0 bytes"
    try:
        import cv2

        arr = np.frombuffer(raw, dtype=np.uint8)
        im = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if im is not None:
            mx = int(im.max())
            mn = int(im.min())
            shape = getattr(im, "shape", ())
            likely_black = mx < 8 and mn == 0
            return likely_black, f"PNG decod. shape={shape} max={mx} min={mn}"
        mx = int(arr.max()) if arr.size else 0
        likely_black = mx < 8
        return likely_black, f"nao-PNG len={len(arr)} max={mx}"
    except Exception as e:
        return True, f"erro decode: {e}"


def offset_xy_perpendicular_to_approach(
    obs_xy: tuple[float, float],
    start_xy: tuple[float, float],
    lateral_m: float,
) -> tuple[float, float]:
    """Desvia start no plano XY ao longo de um vetor perpendicular a (start - obs)."""
    if lateral_m == 0.0:
        return start_xy
    vx = start_xy[0] - obs_xy[0]
    vy = start_xy[1] - obs_xy[1]
    vlen = math.hypot(vx, vy)
    if vlen < 1e-6:
        vx, vy, vlen = 1.0, 0.0, 1.0
    else:
        vx /= vlen
        vy /= vlen
    px, py = -vy, vx
    return (start_xy[0] + px * lateral_m, start_xy[1] + py * lateral_m)


def save_image(response, out_path: Path, is_float: bool = False):
    """Salva resposta de imagem do AirSim como PNG ou PFM.

    Com compress=False (ex.: segmentação), image_data_uint8 é BGR888 em bruto — tem de se
    usar cv2.imwrite; gravar os bytes directamente produz ficheiros .png inválidos.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if is_float:
        arr = airsim.get_pfm_array(response)
        airsim.write_pfm(str(out_path), arr)
        return

    raw = response.image_data_uint8
    if not raw:
        return

    # Resposta comprimida (PNG / JPEG) vinda do simulador
    if len(raw) >= 8 and raw[:8] == b"\x89PNG\r\n\x1a\n":
        out_path.write_bytes(raw)
        return
    if len(raw) >= 2 and raw[:2] == b"\xff\xd8":
        out_path.write_bytes(raw)
        return

    w = int(getattr(response, "width", 0) or 0)
    h = int(getattr(response, "height", 0) or 0)
    if w > 0 and h > 0 and len(raw) == w * h * 3:
        arr = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))
        cv2.imwrite(str(out_path), arr)
        return
    if w > 0 and h > 0 and len(raw) == w * h * 4:
        arr = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 4))
        cv2.imwrite(str(out_path), arr)
        return

    airsim.write_file(str(out_path), raw)


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

    _seg_black_help_printed: bool = False
    _rgb_seg_resolution_warned: bool = False

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
            # Segmentação com compress=True (JPEG) costuma gerar PNG preto / IDs corrompidos no AirSim.
            compress = (not is_float) and key != "seg"
            requests.append(
                airsim.ImageRequest(
                    self.cfg.camera_name,
                    img_type_int,
                    pixels_as_float=is_float,
                    compress=compress,
                )
            )
        return requests

    @staticmethod
    def _bearing_deg_obs_to_target_xy(
        obs_xy: tuple[float, float], target_xy: tuple[float, float]
    ) -> float:
        """Bearing NED em graus (0° = Norte/+X, 90° = Leste/+Y), alinhado ao yaw AirSim rotateToYawAsync."""
        dx = target_xy[0] - obs_xy[0]
        dy = target_xy[1] - obs_xy[1]
        if math.hypot(dx, dy) < 0.05:
            return 0.0
        return math.degrees(math.atan2(dy, dx))

    def _observer_yaw_for_rotate_mode(
        self,
        obs_xy: tuple[float, float],
        intruder_xy: tuple[float, float],
        offset_deg: float,
    ) -> float:
        """Com --rotate-observer: yaw = bearing(Observer → intruso) + offset do cenario (FOV típico 120°)."""
        return self._bearing_deg_obs_to_target_xy(obs_xy, intruder_xy) + float(offset_deg)

    def _set_observer_yaw_deg(self, yaw_deg: float) -> None:
        """Alinha o yaw do Observer.

        A API AirSim (Python client) usa **graus** para yaw e para margin em rotateToYawAsync
        (margin padrao no client = 5). Passar radianos + margin 0.03 faz o comando quase nunca
        convergir e o .join() trava ate o timeout — especialmente no 2º cenario.
        """
        vn = self.cfg.observer_vehicle_name
        if not vn or vn == self.cfg.vehicle_name:
            return
        try:
            fut = self.client.rotateToYawAsync(
                float(yaw_deg),
                timeout_sec=25.0,
                margin=5.0,
                vehicle_name=vn,
            )
            self._join_future_limited(fut, wall_timeout_sec=30.0, label="rotateToYawAsync(Observer)")
        except Exception as e:
            print(f"[AVISO] rotateToYawAsync(Observer): {e}")

    def _join_future_limited(self, fut, wall_timeout_sec: float = 35.0, label: str = "async") -> None:
        """Evita bloqueio indefinido se o sim nao completar o future."""
        done = threading.Event()
        err: list[BaseException] = []

        def _run():
            try:
                fut.join()
            except BaseException as e:
                err.append(e)
            finally:
                done.set()

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        if not done.wait(timeout=wall_timeout_sec):
            print(f"  [AVISO] {label} excedeu {wall_timeout_sec:.0f}s — cancelando tarefa e seguindo.")
            for v_name in set(filter(None, [self.cfg.vehicle_name, self.cfg.observer_vehicle_name])):
                try:
                    self.client.cancelLastTask(vehicle_name=v_name)
                except Exception:
                    pass
        elif err:
            print(f"  [AVISO] {label}: {err[0]}")

    def _prep_vehicles_between_scenarios(self) -> None:
        """Cancela tarefas e paira ambos os veiculos antes de cada cenario."""
        names = set(filter(None, [self.cfg.vehicle_name, self.cfg.observer_vehicle_name]))
        for v_name in names:
            try:
                self.client.cancelLastTask(vehicle_name=v_name)
            except Exception:
                pass
        time.sleep(0.15)
        for v_name in names:
            try:
                self.client.enableApiControl(True, vehicle_name=v_name)
                fut = self.client.hoverAsync(vehicle_name=v_name)
                self._join_future_limited(fut, wall_timeout_sec=12.0, label=f"hoverAsync({v_name})")
            except Exception as e:
                print(f"  [AVISO] prep veiculo '{v_name}': {e}")
        time.sleep(0.25)

    def _save_frame(self, responses, frame_idx: int, run_dir: Path):
        """Salva todos os tipos de imagem de um frame."""
        for resp, key in zip(responses, self.cfg.image_types):
            img_type_int, subdir, _ = IMAGE_TYPE_MAP[key]
            is_float = img_type_int in FLOAT_IMAGE_TYPES
            ext = ".pfm" if is_float else ".png"
            out_path = run_dir / subdir / f"frame_{frame_idx:06d}{ext}"
            if key == "seg" and frame_idx == 0 and resp is not None:
                bad, msg = diagnose_segmentation_response(resp)
                print(f"  [diag seg] {msg}")
                if bad and not self.__class__._seg_black_help_printed:
                    self.__class__._seg_black_help_printed = True
                    print(
                        "  [AVISO] Segmentacao vindo preta/vazia do simulador (nao e o gravador PNG).\n"
                        "    1) Copie para Documents/AirSim/settings.json o bloco \"SegmentationSettings\" de\n"
                        "       config/cosys_airsim_settings.json (InitMethod: CommonObjectsRandomIDs).\n"
                        "    2) Unreal: Project Settings → Rendering → Custom Depth-Stencil Pass = Enabled.\n"
                        "    3) No mapa Cosys, confirme que meshes/landscape entram na pipeline de segmentacao AirSim.\n"
                        "    4) Reinicie o ambiente Unreal por completo apos mudar settings.json."
                    )
            save_image(resp, out_path, is_float=is_float)

        if frame_idx == 0 and len(responses) == len(self.cfg.image_types):
            scene_wh: tuple[int, int] | None = None
            seg_wh: tuple[int, int] | None = None
            for resp, key in zip(responses, self.cfg.image_types):
                if resp is None:
                    continue
                w = int(getattr(resp, "width", 0) or 0)
                h = int(getattr(resp, "height", 0) or 0)
                if w <= 0 or h <= 0:
                    continue
                if key == "scene":
                    scene_wh = (w, h)
                elif key == "seg":
                    seg_wh = (w, h)
            if (
                scene_wh
                and seg_wh
                and scene_wh != seg_wh
                and not self.__class__._rgb_seg_resolution_warned
            ):
                self.__class__._rgb_seg_resolution_warned = True
                print(
                    f"  [AVISO] Resolucao Scene (RGB) {scene_wh[0]}x{scene_wh[1]} != "
                    f"Segmentacao {seg_wh[0]}x{seg_wh[1]}.\n"
                    "    Caixas YOLO a partir da seg assumem o mesmo FOV que o RGB; com resolucoes/FOV diferentes "
                    "o drone desloca-se entre imagens.\n"
                    "    Em Documents/AirSim/settings.json, em Vehicles.Observer.Cameras.front_center.CaptureSettings, "
                    "garanta um bloco ImageType 5 com a mesma Width, Height e FOV_Degrees que ImageType 0 "
                    "(veja config/cosys_airsim_settings.json). Reinicie o Unreal."
                )

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

    def _attach_camera_telemetry(self, tele: dict, responses: list) -> None:
        """Preenche pose da câmera alinhada ao frame (ImageResponse) + FOV para labels por projeção 3D→2D."""
        empty = {
            "img_w": "",
            "img_h": "",
            "cam_fov_deg": "",
            "cam_pos_x": "",
            "cam_pos_y": "",
            "cam_pos_z": "",
            "cam_qw": "",
            "cam_qx": "",
            "cam_qy": "",
            "cam_qz": "",
        }
        tele.update(empty)

        fov = self.cfg.camera_fov_fallback_deg
        try:
            ci = self.client.simGetCameraInfo(
                self.cfg.camera_name, vehicle_name=self.cfg.observer_vehicle_name
            )
            fv = getattr(ci, "fov", None)
            if fv is not None and float(fv) > 0.5:
                fov = float(fv)
        except Exception:
            pass
        tele["cam_fov_deg"] = fov

        if not responses:
            return

        types = self.cfg.image_types
        try:
            ri = types.index("scene")
        except ValueError:
            ri = 0
        r = responses[ri] if ri < len(responses) else responses[0]
        w, h = getattr(r, "width", 0), getattr(r, "height", 0)
        if w and h:
            tele["img_w"] = int(w)
            tele["img_h"] = int(h)
        cp = getattr(r, "camera_position", None)
        co = getattr(r, "camera_orientation", None)
        if cp is None or co is None:
            return
        try:
            if co.containsNan():
                return
        except Exception:
            pass
        tele["cam_pos_x"] = cp.x_val
        tele["cam_pos_y"] = cp.y_val
        tele["cam_pos_z"] = cp.z_val
        tele["cam_qw"] = co.w_val
        tele["cam_qx"] = co.x_val
        tele["cam_qy"] = co.y_val
        tele["cam_qz"] = co.z_val

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

    def _print_progress(self, done: int, total: int):
        """Barra de progresso simples no terminal."""
        if total <= 0:
            return
        width = 30
        ratio = max(0.0, min(1.0, done / total))
        filled = int(width * ratio)
        bar = "#" * filled + "-" * (width - filled)
        print(f"[{bar}] {done}/{total} ({ratio*100:5.1f}%)")

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
        for child in run_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

        # Libera comandos pendentes do cenario anterior (evita 2º run preso em Async antigo)
        self._prep_vehicles_between_scenarios()

        # 1. Ambiente
        self._apply_environment(scenario)

        # 2. Ler posicao REAL do Observer no simulador (nao usar cfg.observer_pos)
        obs_real = self._get_global_position(cfg.observer_vehicle_name)
        print(f"  -> Observer posicao atual (global): ({obs_real[0]:.1f}, {obs_real[1]:.1f}, {obs_real[2]:.1f})")

        # 3. Calcular posicao alvo do intruso relativa ao Observer REAL
        if cfg.rotate_observer_for_geometry:
            start_pos_global = intruder_start_global_ned(
                obs_real,
                scenario.distance_m,
                scenario.azimuth_deg,
                scenario.altitude_offset_m,
            )
            print(
                f"  -> Modo --rotate-observer: intruso bearing {scenario.azimuth_deg:.1f}° NED (azimuth_deg do cenario); "
                f"yaw Observer = bearing→intruso + offset {scenario.rotate_observer_offset_deg:+.1f}°."
            )
        else:
            start_pos_global = scenario.start_position_ned(observer_ned=obs_real)
        if cfg.vertical_separation_ned_m != 0.0:
            start_pos_global = (
                start_pos_global[0],
                start_pos_global[1],
                start_pos_global[2] + cfg.vertical_separation_ned_m,
            )
            print(
                f"  -> Separacao vertical seguranca: +{cfg.vertical_separation_ned_m:.1f} m NED no Z do intruso "
                f"(intruso {cfg.vertical_separation_ned_m:.1f} m abaixo do plano do cenario)."
            )
        if cfg.approach_lateral_offset_m > 0.0:
            sx, sy = offset_xy_perpendicular_to_approach(
                (obs_real[0], obs_real[1]),
                (start_pos_global[0], start_pos_global[1]),
                cfg.approach_lateral_offset_m,
            )
            start_pos_global = (sx, sy, start_pos_global[2])
            print(
                f"  -> Desvio lateral (miss no plano XY): {cfg.approach_lateral_offset_m:.1f} m — "
                f"a reta de voo nao passa pelo ponto do observador (reduz colisao)."
            )
        target_z_global = start_pos_global[2]

        if cfg.rotate_observer_for_geometry:
            yaw_t = self._observer_yaw_for_rotate_mode(
                (obs_real[0], obs_real[1]),
                (start_pos_global[0], start_pos_global[1]),
                scenario.rotate_observer_offset_deg,
            )
            print(f"  -> Yaw Observer (face ao ponto de partida do intruso): {yaw_t:.1f}°")
            self._set_observer_yaw_deg(yaw_t)
            time.sleep(0.4)

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

        if cfg.rotate_observer_for_geometry:
            yaw_t = self._observer_yaw_for_rotate_mode(
                (obs_real[0], obs_real[1]),
                (intr_real[0], intr_real[1]),
                scenario.rotate_observer_offset_deg,
            )
            self._set_observer_yaw_deg(yaw_t)
            time.sleep(0.3)

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
        v_obs = cfg.observer_vehicle_name
        dz_level = abs(intr_real[2] - obs_real[2])
        use_avoid = (
            bool(v_obs)
            and v_obs != cfg.vehicle_name
            and cfg.intruder_avoid_pass_m > 0.01
            and abs(cfg.intruder_avoid_altitude_ned_m) > 0.01
        )
        if use_avoid and dz_level > cfg.intruder_avoid_same_level_tol_m:
            use_avoid = False
            print(
                f"  -> Evita-colisao por altitude: desligado (|dz| obs-intr = {dz_level:.1f} m > "
                f"tol {cfg.intruder_avoid_same_level_tol_m:.1f} m — ja nao estao ao mesmo nivel)."
            )
        if use_avoid:
            d_pass = cfg.intruder_avoid_pass_m
            dz_avoid = cfg.intruder_avoid_altitude_ned_m
            p_turn_global = (
                obs_real[0] + nx * d_pass,
                obs_real[1] + ny * d_pass,
                target_z_global,
            )
            past_global = (
                obs_real[0] - nx * overshoot,
                obs_real[1] - ny * overshoot,
                target_z_global,
            )
            past_alt_global = (
                past_global[0],
                past_global[1],
                target_z_global + dz_avoid,
            )
            loc_turn = self._to_local(cfg.vehicle_name, p_turn_global)
            loc_past_alt = self._to_local(cfg.vehicle_name, past_alt_global)
            print(
                f"  -> Evita-colisao (mesmo nivel): 1) ate ~{d_pass:.0f} m no eixo; "
                f"2) passa atras com delta Z NED {dz_avoid:+.1f} m (so altitude, sem desvio lateral)."
            )
            self.client.moveToPositionAsync(
                loc_turn[0], loc_turn[1], loc_turn[2],
                velocity=scenario.speed_ms,
                timeout_sec=600,
                drivetrain=airsim.DrivetrainType.ForwardOnly,
                yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0),
                vehicle_name=cfg.vehicle_name,
            ).join()
            self.client.moveToPositionAsync(
                loc_past_alt[0], loc_past_alt[1], loc_past_alt[2],
                velocity=scenario.speed_ms,
                timeout_sec=600,
                drivetrain=airsim.DrivetrainType.ForwardOnly,
                yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0),
                vehicle_name=cfg.vehicle_name,
            )
        else:
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

        if cfg.rotate_observer_for_geometry:
            obs_g = self._get_global_position(cfg.observer_vehicle_name)
            intr_g = self._get_global_position(cfg.vehicle_name)
            yaw_t = self._observer_yaw_for_rotate_mode(
                (obs_g[0], obs_g[1]),
                (intr_g[0], intr_g[1]),
                scenario.rotate_observer_offset_deg,
            )
            self._set_observer_yaw_deg(yaw_t)
            time.sleep(0.3)

        # 7. Loop de captura (NAO depende de future._set_flag)
        requests = self._build_image_requests()
        telemetry_rows = []
        frame_idx = 0
        last_capture = time.perf_counter()
        stall_count = 0
        prev_rel_dot = None
        prev_dist = None

        print(f"  -> Capturando {len(self.cfg.image_types)} tipo(s) de imagem...")

        while True:
            now = time.perf_counter()
            if now - last_capture < cfg.capture_interval_s:
                time.sleep(0.005)
                continue
            last_capture = now

            # Imagens da camera do Observer
            responses = []
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
            self._attach_camera_telemetry(tele, responses)

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

            # Anti-stall: evita cenarios presos (ex.: rel_dot/dist quase constantes por muito tempo)
            if prev_rel_dot is not None and prev_dist is not None:
                rel_delta = abs(rel_dot - prev_rel_dot)
                dist_delta = abs(dist - prev_dist)
                if rel_delta < 0.05 and dist_delta < 0.05:
                    stall_count += 1
                else:
                    stall_count = 0
            prev_rel_dot = rel_dot
            prev_dist = dist

            if stall_count >= 120:  # ~12s com capture_interval=0.1
                print("  [AVISO] Cenario aparenta travado (movimento estagnado). Encerrando cenario.")
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
            fieldnames: list[str] = list(telemetry_rows[0].keys())
            for row in telemetry_rows[1:]:
                for k in row:
                    if k not in fieldnames:
                        fieldnames.append(k)
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(telemetry_rows)
            print(f"  -> {frame_idx} frames | telemetria: {csv_path}")
        else:
            print("  [AVISO] Nenhum frame capturado.")

        # Retornar drones para posicoes de spawn (settings.json)
        # Para evitar colisao por mudanca brusca de altitude:
        #  1) move em XY mantendo o Z atual (nivel atual do cenario)
        #  2) depois, ja com XY no ponto, ajusta Z para a altitude de decolagem
        target_z = -cfg.takeoff_altitude_m
        print("  -> Retornando drones para posicoes iniciais (XY sem trocar Z)...")

        vehicle_names = list(set(filter(None, [cfg.vehicle_name, cfg.observer_vehicle_name])))
        current_z: dict[str, float] = {}
        for v_name in vehicle_names:
            try:
                _gx, _gy, gz = self._get_global_position(v_name)
                current_z[v_name] = float(gz)
            except Exception:
                current_z[v_name] = float(target_z)

        # Etapa 1: XY em Z atual
        futures = []
        for v_name in vehicle_names:
            spawn = self._home_offsets.get(v_name, (0.0, 0.0, 0.0))
            local_target = self._to_local(v_name, (spawn[0], spawn[1], current_z[v_name]))
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

        # Etapa 2: Z para altitude de decolagem
        print("  -> Ajustando altitude (Z) para posicao de decolagem...")
        futures = []
        for v_name in vehicle_names:
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
        """Executa todos os cenários em sequência (opcionalmente para múltiplas câmeras)."""
        camera_names = self.cfg.camera_names or [self.cfg.camera_name]
        original_output_dir = self.cfg.output_dir
        total = len(scenarios) * len(camera_names)
        done = 0
        print(f"\n[INFO] Iniciando {len(scenarios)} cenario(s) x {len(camera_names)} camera(s) = {total} execucoes...")
        self._print_progress(done, total)

        for cam in camera_names:
            self.cfg.camera_name = cam
            self.cfg.output_dir = original_output_dir / cam
            self.cfg.output_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n[INFO] Camera: {cam} | Saida: {self.cfg.output_dir}")

            for i, scenario in enumerate(scenarios, 1):
                try:
                    self.run_scenario(scenario, run_idx=i)
                except KeyboardInterrupt:
                    print("\n[INTERROMPIDO] Encerrando experimento.")
                    return
                except Exception as e:
                    print(f"[ERRO] Cenario '{scenario.name}' (camera {cam}): {e}")
                    # Tenta recuperar sem reset (que mata a conexao)
                    for v_name in set(filter(None, [self.cfg.vehicle_name, self.cfg.observer_vehicle_name])):
                        try:
                            self.client.enableApiControl(True, vehicle_name=v_name)
                            self.client.hoverAsync(vehicle_name=v_name).join()
                        except Exception:
                            pass
                finally:
                    done += 1
                    self._print_progress(done, total)

        self.cfg.output_dir = original_output_dir
        print(f"\n[CONCLUIDO] {done}/{total} execucoes | Dataset: {original_output_dir.resolve()}")


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
        "--first-n", type=int, default=None, metavar="N",
        help="Executa os primeiros N cenarios na ordem de scenarios.py (ex: 3 para teste rapido no novo ambiente).",
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
        help="Metros além do observador no eixo de aproximação para terminar a captura "
             "(rel_dot < -stop-dist). Aumente (ex.: 15–25) para gravar depois de cruzar.",
    )
    p.add_argument(
        "--vertical-separation", type=float, default=1.0, metavar="M",
        help="Separacao vertical minima (m): desloca o intruso para BAIXO no NED AirSim "
             "(Z mais positivo). Padrao 1.0 m evita mesmo plano horizontal com o observador. Use 0 para desligar.",
    )
    p.add_argument(
        "--rotate-observer", action="store_true",
        help="Intruso em --fixed-intruder-azimuth; yaw do Observer = bearing→intruso + rotate_observer_offset_deg no cenario.",
    )
    p.add_argument(
        "--fixed-intruder-azimuth", type=float, default=0.0,
        help="Sem --rotate-observer: azimute inicial do intruso. Com --rotate-observer: ignorado (usa azimuth_deg do cenario).",
    )
    p.add_argument(
        "--approach-lateral-offset", type=float, default=-1.0, metavar="M",
        help="Metros: desloca o ponto de partida perpendicular à linha observador->partida, para a rota "
             "passar ao lado do observador (evita colisao no plano horizontal). "
             "Padrao -1 = automatico: 4 m se --rotate-observer, senao 0. Use 0 para forcar sem desvio.",
    )
    p.add_argument(
        "--avoid-pass-m", type=float, default=10.0, metavar="M",
        help="Com Observer+intruso distintos: voo ate ~M m do obs no eixo; depois (se coplanares) passa atras "
             "só mudando Z (--avoid-altitude-ned-m).",
    )
    p.add_argument(
        "--avoid-altitude-ned-m", type=float, default=5.0, metavar="M",
        help="Delta Z NED no segundo segmento (apos --avoid-pass-m). Positivo = intruso desce no mundo. "
             "So usado se |z_obs-z_intr| <= --avoid-same-level-tol-m.",
    )
    p.add_argument(
        "--avoid-same-level-tol-m", type=float, default=3.0, metavar="M",
        help="Só aplica evita-colisao por altitude se |z observador - z intruso| <= M m (mesmo nivel).",
    )
    p.add_argument(
        "--avoid-lateral-m", type=float, default=None, metavar="M",
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--no-avoid", action="store_true",
        help="Linha reta ate atras do observador (sem segundo segmento de evita-colisao).",
    )
    p.add_argument(
        "--capture-interval", type=float, default=0.1,
        help="Segundos entre capturas (padrão 0.1 ≈ 10 Hz). Valores menores = mais frames e menor "
             "espaçamento ao longo da aproximação (ex: 0.05 ≈ 20 Hz, 0.033 ≈ 30 Hz).",
    )
    p.add_argument(
        "--dense-capture", action="store_true",
        help="Atalho: força intervalo 0.05 s (~20 Hz). Sobrescreve --capture-interval.",
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
        "--camera-fov-fallback", type=float, default=120.0, metavar="DEG",
        help="FOV (graus) gravado em telemetry.csv quando a API nao devolver FOV valido; "
             "deve coincidir com FOV_Degrees da CaptureSettings da cena no settings.json (padrao 120).",
    )
    p.add_argument(
        "--camera", type=str, default="front_center",
        help="Nome da câmera de captura (padrão: front_center).",
    )
    p.add_argument(
        "--cameras", nargs="+", type=str, default=None,
        help="Lista de câmeras para executar em batch (ex: front_center front_narrow_hd).",
    )
    p.add_argument(
        "--ip", type=str, default="127.0.0.1",
        help="IP do host Cosys-AirSim (padrão: 127.0.0.1).",
    )
    p.add_argument(
        "--max-frames", type=int, default=None, metavar="N",
        help="Limite de frames por cenario (teste rapido). Omita para gravar ate o fim da aproximacao.",
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
    if getattr(args, "avoid_lateral_m", None) is not None:
        print(
            "[AVISO] --avoid-lateral-m ja nao e usado; evita-colisao e so por altitude "
            "(--avoid-altitude-ned-m) quando ao mesmo nivel (--avoid-same-level-tol-m)."
        )
    if args.dense_capture:
        args.capture_interval = 0.05

    if args.approach_lateral_offset < -0.5:
        approach_lateral_m = 4.0 if args.rotate_observer else 0.0
    else:
        approach_lateral_m = max(0.0, args.approach_lateral_offset)

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
        rot_note = (
            " | --rotate-observer: yaw Obs = bearing→intruso + rotate_observer_offset_deg"
            if args.rotate_observer
            else ""
        )
        print(f"\n{'#':>3}  {'Nome':<30} {'Pos Inicial NED (x,y,z)':^30}  Resumo{rot_note}")
        print("─" * 110)
        for i, s in enumerate(all_scenarios, 1):
            if args.rotate_observer:
                pos = intruder_start_global_ned(
                    observer, s.distance_m, s.azimuth_deg, s.altitude_offset_m
                )
            else:
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
    elif args.first_n is not None:
        if args.first_n < 1:
            print("[ERRO] --first-n deve ser >= 1.")
            sys.exit(1)
        scenarios_to_run = all_scenarios[: args.first_n]
        print(f"[INFO] --first-n {args.first_n}: {len(scenarios_to_run)} cenario(s): "
              f"{', '.join(s.name for s in scenarios_to_run)}")
    elif not args.list:
        print("[ERRO] Especifique --scenario <nome>, --scenarios <n1 n2 ...>, --first-n N, --all, --list ou --dry-run.")
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
        camera_names=args.cameras,
        camera_fov_fallback_deg=args.camera_fov_fallback,
        vertical_separation_ned_m=args.vertical_separation,
        rotate_observer_for_geometry=args.rotate_observer,
        fixed_intruder_azimuth_deg=args.fixed_intruder_azimuth,
        approach_lateral_offset_m=approach_lateral_m,
        intruder_avoid_pass_m=0.0 if args.no_avoid else args.avoid_pass_m,
        intruder_avoid_altitude_ned_m=0.0 if args.no_avoid else args.avoid_altitude_ned_m,
        intruder_avoid_same_level_tol_m=args.avoid_same_level_tol_m,
    )

    if approach_lateral_m > 0:
        print(
            f"[INFO] approach_lateral_offset_m={approach_lateral_m} — rota deslocada no plano XY "
            "(minima distancia horizontal ao observador)."
        )

    controller = ExperimentController(cfg)
    controller.connect()

    # Com --cameras, sempre run_all (1+ cenarios ou varias cameras); senao 1 cenario usa run_scenario direto.
    if len(scenarios_to_run) == 1 and args.cameras is None:
        controller.run_scenario(scenarios_to_run[0], run_idx=1)
    else:
        controller.run_all(scenarios_to_run)


if __name__ == "__main__":
    main()
