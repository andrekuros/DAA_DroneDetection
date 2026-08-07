"""
run_gnss_denial_experiment.py — Experimento de deriva inercial sem GNSS
=======================================================================
Orquestra um voo autonomo no PX4 SITL (ligado ao Cosys-AirSim) registrando, a
20 Hz, a estimativa do EKF2 contra o ground truth do simulador. Em um ponto
definido da trajetoria o GNSS e negado (EKF2_GPS_CTRL=0), deixando o estimador
em dead-reckoning inercial. O resultado e um CSV por voo + figuras de deriva.

Perfil de voo: arma -> sobe a 60 m por offboard -> voa 250 m no eixo X -> pousa.
A negacao ocorre quando o veiculo percorreu --deny-at-m no sentido do voo.

O voo acontece DENTRO do canion urbano: a 60 m as torres do CitySample ainda passam
muito acima, e colisao foi observada. Cada impacto perturba IMU e dinamica na janela
medida, entao o CSV tem `collided`/`collision_object` e o voo aborta ao colidir.
A campanha alterna o sentido (--direction) para repetir sempre o mesmo trecho.

Dependencias:
    pip install mavsdk cosysairsim matplotlib numpy

Pre-requisitos:
    1. config/cosys_airsim_px4_settings.json copiado para Documents/AirSim/settings.json
    2. Ambiente Unreal com Cosys-AirSim rodando
    3. PX4 SITL rodando no WSL2 (alvo none_iris, PX4_SIM_HOSTNAME=127.0.0.1)

Uso:
    # Estagio 1 — valida imports/config, sem simulador
    python tools/run_gnss_denial_experiment.py --dry-run

    # Estagio 2 — conecta e loga 10 s parado, sem armar
    python tools/run_gnss_denial_experiment.py --no-fly --watch-s 10

    # Estagio 3 — voo de referencia, GNSS sempre ligado
    python tools/run_gnss_denial_experiment.py --deny-at-m -1

    # Estagio 4 — run experimental: nega GNSS aos 15 m
    python tools/run_gnss_denial_experiment.py --deny-at-m 15

Coordenadas NED: X=Norte, Y=Leste, Z=para baixo (Z negativo = acima do solo).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from airsim_gt import AirSimGroundTruth  # noqa: E402
from px4_link import DEFAULT_SYSTEM_ADDRESS, PX4Link  # noqa: E402

# As 7 colunas pedidas no protocolo, mais o minimo para tornar o dado analisavel
# sem depender de metadados externos.
CSV_FIELDS = [
    "timestamp",       # epoch wall-clock (s)
    "px4_x", "px4_y", "px4_z",       # estimativa EKF2, NED local do PX4
    "cosys_x", "cosys_y", "cosys_z",  # ground truth do simulador, NED global (bruto, no tick)
    "t_mono",          # segundos desde o inicio do log (eixo dos graficos)
    "gps_denied",      # 0 antes do corte, 1 depois
    "err_x", "err_y", "err_z", "err_norm",  # erro alinhado E compensado em latencia
    "px4_age_s",       # idade da amostra do PX4 no instante do tick
    "err_raw_norm",    # erro sem compensar latencia (para auditar a correcao)
    "t_since_denial",  # segundos desde o corte de GNSS (<0 antes); eixo da lei de deriva
    # Fatores para o factor graph (reprocessamento offline da mesma janela)
    "imu_ax", "imu_ay", "imu_az", "imu_gx", "imu_gy", "imu_gz",
    "baro_alt_m", "baro_pressure", "mag_x", "mag_y", "mag_z",
    # Observabilidade de scan-matching: quanta estrutura o LiDAR enxerga
    "lidar_points", "lidar_mean_range_m", "lidar_min_range_m",
    # Validade: impacto contra a cena contamina IMU e dinamica
    "collided", "collision_object",
]


@dataclass
class ExperimentConfig:
    """Parametros do experimento (ecoados no meta.json para reprodutibilidade)."""
    vehicle_name: str = "PX4Drone"
    system_address: str = DEFAULT_SYSTEM_ADDRESS
    airsim_ip: str = "127.0.0.1"
    # 60 m NAO esta acima do skyline do CitySample: as torres passam bem disso, e
    # colisao a 50 m foi observada. A escolha e deliberada mesmo assim — o voo fica
    # dentro do canion urbano, que e o ambiente de interesse e onde o LiDAR ve
    # estrutura lateral. O que torna isso viavel nao e a altitude, e sim o vaivem
    # no mesmo corredor (--direction alternado) mais a deteccao de colisao, que
    # aborta e repoe o voo em vez de deixar o impacto contaminar a medida.
    takeoff_alt_m: float = 60.0
    # Voo longo: a lei de deriva se le ao longo de UM voo, em t_since_denial.
    # Negar cedo e voar muito depois amostra a curva inteira; negar tarde num voo
    # curto so mede "quanto de pista sobrou", que nao e a mesma grandeza.
    distance_m: float = 250.0
    deny_at_m: float = 15.0       # negar GNSS apos N metros em +X; <0 desativa
    rate_hz: float = 20.0
    lidar_hz: float = 4.0         # cadencia da nuvem (leitura RPC cara)
    # Latencia de transporte PX4 -> MAVLink -> mavsdk_server -> este processo.
    # Nao e observavel pelo timestamp de chegada (px4_age_s fica em ~2 ms), mas
    # aparece como erro proporcional a velocidade. Medida por minimizacao de RMS
    # num voo de referencia com GNSS ligado: 255 ms nesta bancada.
    # Reestimar com: python tools/plot_gnss_drift.py <run> --estimate-latency
    transport_latency_s: float = 0.255
    out_dir: Path = Path("dataset_gnss_denial")
    run_name: str = ""
    land_at_end: bool = True
    # +1 = Norte, -1 = Sul. A campanha alterna para o veiculo repetir o mesmo
    # corredor validado em vez de avancar para cidade nova a cada voo.
    direction: int = 1
    # Coordenada E (Leste) do corredor validado. None = usa a posicao atual.
    # Fixar em absoluto faz o vaivem repetir SEMPRE a mesma rua, que e o unico
    # jeito de a validacao do corredor valer para todos os voos da campanha.
    corridor_e_m: float | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Logger CSV — escrita incremental
# ─────────────────────────────────────────────────────────────────────────────

class TelemetryLogger:
    """
    Grava uma linha por amostra, com flush imediato.

    Diverge de experiment_controller.py, que acumula tudo em memoria e so grava
    no fim: aqui um Ctrl+C ou uma queda do PX4 no meio do voo precisa deixar o
    CSV integro em disco, porque o voo nao e barato de repetir.
    """

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=CSV_FIELDS)
        self._writer.writeheader()
        self._fh.flush()
        self.n_rows = 0

    def write(self, row: dict) -> None:
        self._writer.writerow(row)
        self._fh.flush()
        self.n_rows += 1

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()


# ─────────────────────────────────────────────────────────────────────────────
# Loop de amostragem
# ─────────────────────────────────────────────────────────────────────────────

async def sample_loop(
    link: PX4Link,
    gt: AirSimGroundTruth,
    logger: TelemetryLogger,
    cfg: ExperimentConfig,
    state: dict,
    stop_event: asyncio.Event,
) -> None:
    """
    Amostra PX4 + AirSim a cfg.rate_hz ate stop_event, gravando cada linha.

    Tambem dispara a negacao de GNSS quando o ground truth indica que o veiculo
    percorreu cfg.deny_at_m em +X. O gatilho usa ground truth de proposito: se
    usasse a estimativa do PX4, o ponto de corte se deslocaria justamente por
    causa da deriva que estamos medindo.
    """
    period = 1.0 / cfg.rate_hz
    t0 = time.monotonic()
    # A nuvem do LiDAR e a leitura mais cara do tick; amostrar mais devagar mantem
    # a cadencia de telemetria estavel sem perder o sinal de estrutura, que varia
    # bem mais lentamente que a posicao.
    lidar_every = max(1, int(round(cfg.rate_hz / max(cfg.lidar_hz, 0.1))))
    last_sensors = None
    tick_i = 0

    while not stop_event.is_set():
        tick = time.monotonic()
        want_lidar = (tick_i % lidar_every) == 0
        tick_i += 1

        # RPC do AirSim e bloqueante -> thread separada, senao trava o event loop.
        # read_all_async usa UMA thread dedicada: o cliente RPC nao e thread-safe e
        # precisa de event loop por thread (ver airsim_gt._ensure_pool).
        gt_s, sen = await gt.read_all_async(want_lidar)
        if want_lidar or last_sensors is None:
            last_sensors = sen
        else:
            # Repete a ultima leitura valida de LiDAR nos ticks em que nao foi lida.
            sen.lidar_points = last_sensors.lidar_points
            sen.lidar_mean_range_m = last_sensors.lidar_mean_range_m
            sen.lidar_min_range_m = last_sensors.lidar_min_range_m
        px4_s = link.latest

        if state.get("origin_offset") is None and gt_s.valid and px4_s.valid:
            # A origem NED do PX4 (init do EKF) nao coincide com a do AirSim
            # (spawn). Captura o offset na primeira amostra valida e usa dele em
            # diante para que err_* meça deriva, e nao a diferenca de referencial.
            state["origin_offset"] = (
                gt_s.x_m - px4_s.north_m,
                gt_s.y_m - px4_s.east_m,
                gt_s.z_m - px4_s.down_m,
            )
            state["start_x"] = gt_s.x_m
            print(f"[INFO] Offset de origem PX4->AirSim: {state['origin_offset']}")

        ox, oy, oz = state.get("origin_offset") or (0.0, 0.0, 0.0)

        # Compensacao de latencia. A amostra do PX4 chegou em px4_s.t_wall, mas o
        # ground truth foi lido agora: comparar os dois crus transforma a defasagem
        # em erro proporcional a velocidade (medido: ~0,2 s => 3,4 m a 12 m/s, mais
        # do que a propria deriva que queremos observar). Retrocedemos o ground
        # truth ate o instante da amostra do PX4 usando a velocidade verdadeira,
        # exato em primeira ordem para o intervalo curto envolvido.
        age_cache = max(0.0, gt_s.t_wall - px4_s.t_wall) if (gt_s.valid and px4_s.valid) else 0.0
        age_cache = min(age_cache, 1.0)  # mais velho que isso indica stream travado
        age = age_cache + cfg.transport_latency_s
        gx = gt_s.x_m - gt_s.vx_ms * age
        gy = gt_s.y_m - gt_s.vy_ms * age
        gz = gt_s.z_m - gt_s.vz_ms * age

        ex = gx - (px4_s.north_m + ox)
        ey = gy - (px4_s.east_m + oy)
        ez = gz - (px4_s.down_m + oz)

        rx = gt_s.x_m - (px4_s.north_m + ox)
        ry = gt_s.y_m - (px4_s.east_m + oy)
        rz = gt_s.z_m - (px4_s.down_m + oz)

        logger.write({
            "timestamp": time.time(),
            "px4_x": px4_s.north_m,
            "px4_y": px4_s.east_m,
            "px4_z": px4_s.down_m,
            "cosys_x": gt_s.x_m,
            "cosys_y": gt_s.y_m,
            "cosys_z": gt_s.z_m,
            "t_mono": tick - t0,
            "gps_denied": 1 if link.gps_denied else 0,
            "err_x": ex,
            "err_y": ey,
            "err_z": ez,
            "err_norm": math.sqrt(ex * ex + ey * ey + ez * ez),
            "px4_age_s": age_cache,
            "err_raw_norm": math.sqrt(rx * rx + ry * ry + rz * rz),
            "t_since_denial": (
                (tick - t0) - state["denial_t_mono"]
                if state.get("denial_t_mono") is not None else -1.0
            ),
            "imu_ax": sen.ax, "imu_ay": sen.ay, "imu_az": sen.az,
            "imu_gx": sen.gx, "imu_gy": sen.gy, "imu_gz": sen.gz,
            "baro_alt_m": sen.baro_alt_m, "baro_pressure": sen.baro_pressure,
            "mag_x": sen.mag_x, "mag_y": sen.mag_y, "mag_z": sen.mag_z,
            "lidar_points": sen.lidar_points,
            "lidar_mean_range_m": sen.lidar_mean_range_m,
            "lidar_min_range_m": sen.lidar_min_range_m,
            "collided": sen.collided,
            "collision_object": sen.collision_object,
        })

        # Contato com o solo/cenario antes de decolar nao e colisao de voo: o
        # veiculo nasce encostado no chao (e as vezes num pedestre da cena).
        in_flight = gt_s.valid and (-gt_s.z_m) > 3.0
        if sen.collided and not in_flight:
            sen.collided = 0
            sen.collision_object = ""

        if sen.collided and not state.get("collision_warned"):
            state["collision_warned"] = True
            state["first_collision_t"] = tick - t0
            state["first_collision_object"] = sen.collision_object
            print(f"[AVISO] COLISAO com '{sen.collision_object}' em t={tick - t0:.1f}s "
                  f"(X={gt_s.x_m:.1f}, alt={-gt_s.z_m:.1f} m).")
            # Encerrar aqui e o que impede a cascata: um voo que segue apos colidir
            # termina no ar, e o run seguinte comeca ja colidindo em altitude — foi
            # exatamente o que contaminou o rep02. O trecho ate aqui continua valido
            # e e o que a analise usa.
            print("[INFO] Abortando o voo para pousar e liberar a bancada para o proximo run.")
            stop_event.set()
            return

        # Gatilho da negacao de GNSS.
        if (
            cfg.deny_at_m >= 0
            and not link.gps_denied
            and state.get("start_x") is not None
            and gt_s.valid
            and (gt_s.x_m - state["start_x"]) * cfg.direction >= cfg.deny_at_m
        ):
            travelled = (gt_s.x_m - state["start_x"]) * cfg.direction
            print(f"\n[INFO] {travelled:.1f} m percorridos - negando GNSS agora.\n")
            try:
                state["denial"] = await link.deny_gnss()
                state["denial_t_mono"] = tick - t0
                state["denial_x"] = gt_s.x_m
            except Exception as e:
                # Sem negacao nao ha experimento. Continuar geraria um run rotulado
                # "denied" com GNSS ativo do inicio ao fim — dado invalido que passa
                # por valido na agregacao. Melhor abortar e perder o voo.
                print(f"[ERRO] Falha ao negar GNSS: {e}")
                state["denial_failed"] = str(e)
                stop_event.set()
                return

        elapsed = time.monotonic() - tick
        await asyncio.sleep(max(0.0, period - elapsed))


async def flight_profile(link: PX4Link, cfg: ExperimentConfig, stop_event: asyncio.Event) -> None:
    """Perfil de voo completo; sinaliza stop_event ao terminar (ou falhar)."""
    try:
        # arm_and_takeoff ja entra em offboard e sobe ate a altitude alvo.
        await link.arm_and_takeoff(east_m=cfg.corridor_e_m)
        await link.fly_straight_x(distance_m=cfg.distance_m, alt_m=cfg.takeoff_alt_m,
                                  direction=cfg.direction, east_m=cfg.corridor_e_m)
        # Deixa o estimador assentar no ponto final antes de encerrar o log.
        await asyncio.sleep(3.0)
    finally:
        stop_event.set()


# ─────────────────────────────────────────────────────────────────────────────
# Orquestracao
# ─────────────────────────────────────────────────────────────────────────────

async def run(cfg: ExperimentConfig, no_fly: bool, watch_s: float, do_plot: bool) -> int:
    run_name = cfg.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = cfg.out_dir / run_name
    csv_path = run_dir / "telemetry.csv"
    meta_path = run_dir / "meta.json"

    logger = TelemetryLogger(csv_path)
    state: dict = {"origin_offset": None, "start_x": None}
    stop_event = asyncio.Event()
    link = PX4Link(system_address=cfg.system_address, takeoff_alt_m=cfg.takeoff_alt_m)
    gt = AirSimGroundTruth(vehicle_name=cfg.vehicle_name, ip=cfg.airsim_ip)
    status = "ok"

    try:
        # connect() na mesma thread dedicada que fara os RPCs seguintes: o cliente
        # msgpack-rpc guarda estado ligado a thread que o criou.
        await asyncio.get_running_loop().run_in_executor(gt._ensure_pool(), gt.connect)
        await link.connect()
        await link.set_stream_rate(25.0)
        link.start_streams()

        state["health_before"] = await link.health_report()
        state["hgt_ref"] = await link.ensure_baro_height_ref()

        sampler = asyncio.create_task(sample_loop(link, gt, logger, cfg, state, stop_event))

        if no_fly:
            print(f"[INFO] --no-fly: apenas logando por {watch_s:.0f} s (sem armar).")
            await asyncio.sleep(watch_s)
            stop_event.set()
        else:
            await flight_profile(link, cfg, stop_event)

        await sampler
        if state.get("denial_failed"):
            raise RuntimeError(f"Negacao de GNSS falhou: {state['denial_failed']}")

    except KeyboardInterrupt:
        print("\n[INTERROMPIDO] Ctrl+C - encerrando e preservando o CSV.")
        status = "interrupted"
        stop_event.set()
    except Exception as e:
        print(f"\n[ERRO] {type(e).__name__}: {e}")
        status = f"error: {type(e).__name__}: {e}"
        stop_event.set()
    finally:
        # Ordem deliberada: fechar o CSV primeiro. Pouso e plot podem falhar, e
        # nao podem levar junto o dado que ja foi coletado.
        logger.close()
        gt.close()
        print(f"[OK] CSV com {logger.n_rows} linhas: {csv_path}")

        try:
            if link.drone is not None:
                await link.shutdown(land=cfg.land_at_end and not no_fly)
        except Exception as e:
            print(f"[AVISO] Encerramento do PX4 com erro: {e}")

        meta = {
            "run_name": run_name,
            "status": status,
            "config": {**asdict(cfg), "out_dir": str(cfg.out_dir)},
            "rows": logger.n_rows,
            "origin_offset_px4_to_airsim": state.get("origin_offset"),
            "start_x": state.get("start_x"),
            "health_before": state.get("health_before"),
            "hgt_ref": state.get("hgt_ref"),
            "denial": state.get("denial"),
            "denial_t_mono": state.get("denial_t_mono"),
            "denial_x": state.get("denial_x"),
            "collision": {
                "occurred": bool(state.get("collision_warned")),
                "first_t_mono": state.get("first_collision_t"),
                "object": state.get("first_collision_object"),
            },
        }
        meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        print(f"[OK] Metadados: {meta_path}")

        if do_plot and logger.n_rows > 1:
            try:
                from plot_gnss_drift import build_all_figures
                build_all_figures(csv_path, meta_path)
            except Exception as e:
                print(f"[AVISO] Falha ao gerar figuras: {e}")

    return 0 if status == "ok" else 1


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Experimento de deriva da odometria inercial do PX4 sem GNSS"
    )
    ap.add_argument("--dry-run", action="store_true",
                    help="Imprime a configuracao e sai (nao conecta a nada)")
    ap.add_argument("--no-fly", action="store_true",
                    help="Conecta e loga parado, sem armar (validacao da ponte)")
    ap.add_argument("--watch-s", type=float, default=10.0, metavar="S",
                    help="Duracao do log em --no-fly")
    ap.add_argument("--vehicle", default="PX4Drone", metavar="NOME",
                    help="Nome do veiculo no settings.json")
    ap.add_argument("--system-address", default=DEFAULT_SYSTEM_ADDRESS, metavar="URL",
                    help="Endereco MAVSDK do PX4")
    ap.add_argument("--ip", default="127.0.0.1", metavar="IP", help="IP do Cosys-AirSim")
    ap.add_argument("--alt-m", type=float, default=60.0, metavar="M", help="Altitude de decolagem")
    ap.add_argument("--distance-m", type=float, default=250.0, metavar="M",
                    help="Distancia do trecho reto em +X")
    ap.add_argument("--deny-at-m", type=float, default=15.0, metavar="M",
                    help="Nega GNSS apos N metros em +X; use -1 para nunca negar (baseline)")
    ap.add_argument("--rate-hz", type=float, default=20.0, metavar="HZ", help="Taxa de amostragem")
    ap.add_argument("--lidar-hz", type=float, default=4.0, metavar="HZ",
                    help="Cadencia de leitura da nuvem LiDAR (0 desativa)")
    ap.add_argument("--latency-s", type=float, default=0.255, metavar="S",
                    help="Latencia de transporte do PX4, medida no voo de referencia")
    ap.add_argument("--out-dir", type=Path, default=Path("dataset_gnss_denial"), metavar="DIR")
    ap.add_argument("--run-name", default="", metavar="NOME",
                    help="Nome da pasta do run (padrao: timestamp)")
    ap.add_argument("--corridor-e", type=float, default=None, metavar="M",
                    help="Coordenada E do corredor validado (probe_gnss_corridor.py)")
    ap.add_argument("--direction", type=int, default=1, choices=[1, -1], metavar="D",
                    help="Sentido do trecho reto: 1=Norte, -1=Sul (vaivem na campanha)")
    ap.add_argument("--no-land", action="store_true", help="Nao pousa ao final")
    ap.add_argument("--no-plot", action="store_true", help="Nao gera figuras ao final")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig(
        vehicle_name=args.vehicle,
        system_address=args.system_address,
        airsim_ip=args.ip,
        takeoff_alt_m=args.alt_m,
        distance_m=args.distance_m,
        deny_at_m=args.deny_at_m,
        rate_hz=args.rate_hz,
        lidar_hz=args.lidar_hz,
        transport_latency_s=args.latency_s,
        out_dir=args.out_dir,
        run_name=args.run_name,
        land_at_end=not args.no_land,
        direction=args.direction,
        corridor_e_m=args.corridor_e,
    )

    if args.dry_run:
        print("-" * 70)
        print("Experimento GNSS-denied - configuracao (dry-run, nada foi conectado)")
        print("-" * 70)
        for k, v in asdict(cfg).items():
            print(f"  {k:20s} = {v}")
        deny = "nunca (baseline com GNSS)" if cfg.deny_at_m < 0 else f"apos {cfg.deny_at_m:.1f} m em +X"
        print(f"\n  Perfil : armar -> {cfg.takeoff_alt_m:.0f} m -> {cfg.distance_m:.0f} m em +X -> pousar")
        print(f"  Negacao: {deny}")
        print(f"  Saida  : {cfg.out_dir / (cfg.run_name or 'run_<timestamp>')}/telemetry.csv")
        print(f"  Colunas: {', '.join(CSV_FIELDS)}")
        return

    raise SystemExit(asyncio.run(run(cfg, args.no_fly, args.watch_s, not args.no_plot)))


if __name__ == "__main__":
    main()
