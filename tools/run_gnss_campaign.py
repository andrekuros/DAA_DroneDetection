"""
run_gnss_campaign.py — Campanha de repeticoes do experimento GNSS-denied
========================================================================
Roda N voos identicos, reiniciando o PX4 SITL e resetando o Cosys-AirSim entre
cada um, para que todos partam das MESMAS condicoes iniciais. Sem isso os voos
encadeiam: o drone nao volta para a origem, a posicao acumula, e o estado do EKF
carrega historico do voo anterior — o que inviabiliza barras de erro honestas.

Ao final agrega os runs e ajusta a lei de crescimento da deriva.

Pre-requisitos: Unreal + Cosys-AirSim rodando (o PX4 este script gerencia).

Uso:
    # Campanha padrao: 5 repeticoes, nega aos 15 m, voa 250 m
    python tools/run_gnss_campaign.py --reps 5

    # Inclui voos de referencia (GNSS ligado) para o piso de medicao
    python tools/run_gnss_campaign.py --reps 5 --baseline-reps 2

    # So agrega runs ja gravados, sem voar
    python tools/run_gnss_campaign.py --only-aggregate --out-dir dataset_gnss_denial/campaign_01
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path


def _wsl_path(p: Path) -> str:
    """Converte W:\\dir\\file para /mnt/w/dir/file, para o shell do WSL escrever nele."""
    p = p.resolve()
    drive = p.drive.rstrip(":").lower()
    rest = str(p)[len(p.drive):].replace("\\", "/")
    return f"/mnt/{drive}{rest}"

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

PX4_DIR = "~/PX4-Autopilot"
PX4_TARGET = "px4_sitl_default none_iris"
SIM_READY_MARKER = "Simulator connected"
WARMUP_S = 25.0  # ver comentario em px4_start()


# ─────────────────────────────────────────────────────────────────────────────
# Ciclo de vida do PX4
# ─────────────────────────────────────────────────────────────────────────────

def px4_stop() -> None:
    """Mata o PX4 no WSL e espera o processo sumir."""
    subprocess.run(
        ["wsl", "-e", "bash", "-lc", 'pkill -f "build/px4_sitl_default/bin/px4"'],
        capture_output=True, timeout=30,
    )
    for _ in range(20):
        r = subprocess.run(
            ["wsl", "-e", "bash", "-lc", "pgrep -f build/px4_sitl_default/bin/px4"],
            capture_output=True, text=True, timeout=15,
        )
        if not r.stdout.strip():
            return
        time.sleep(0.5)
    print("[AVISO] PX4 nao encerrou no tempo esperado.")


def airsim_reset(ip: str, vehicle: str) -> None:
    """
    Devolve o veiculo ao ponto de spawn. DESLIGADO por padrao — ver aviso.

    ATENCAO: com veiculo PX4, client.reset() quebra a reinicializacao do GPS.
    Medido nesta bancada: apos o reset o PX4 fica em local_position_ok=True mas
    global_position_ok=False e home_position_ok=False permanentemente, e o voo
    nunca chega a armar. Por isso o padrao da campanha e NAO resetar: reiniciar
    so o PX4 ja da um EKF limpo a cada voo, que e o que importa para medir deriva.
    A posicao inicial varia entre voos, mas a deriva e medida a partir do instante
    da negacao, entao isso nao confunde a medida.
    """
    try:
        from airsim_gt import AirSimGroundTruth
        gt = AirSimGroundTruth(vehicle_name=vehicle, ip=ip)
        gt.connect()
        gt.client.reset()
        time.sleep(2.0)
        print("[OK] Cosys-AirSim resetado ao spawn.")
    except Exception as e:
        print(f"[AVISO] Reset do AirSim falhou ({e}); o voo pode nao partir da origem.")


def px4_start(log_path: Path, timeout_s: float = 180.0) -> subprocess.Popen:
    """Sobe o PX4 e espera ele fechar o lockstep com o simulador."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        log_path.unlink()

    # Sem TTY o PX4 redesenha o prompt "pxh>" indefinidamente: o log cru chegou a
    # 322 MB em poucos minutos nesta bancada. Como o readiness check reabre o
    # arquivo em loop, isso satura disco e CPU e a telemetria do PX4 deixa de
    # fluir — o sintoma aparece como "sem mensagens de health", sem relacao
    # aparente com a causa. Filtrar na origem mantem o log em alguns KB.
    filt = (
        f"cd {PX4_DIR} && PX4_SIM_HOSTNAME=127.0.0.1 make {PX4_TARGET} 2>&1 "
        r"| tr '\r' '\n' "
        r"| grep --line-buffered -aE 'Simulator connected|mode: |ERROR|FATAL' "
        f"> {shlex.quote(str(_wsl_path(log_path)))}"
    )
    proc = subprocess.Popen(["wsl", "-e", "bash", "-lc", filt])

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            if SIM_READY_MARKER in log_path.read_text(encoding="utf-8", errors="replace"):
                # "Simulator connected" so diz que o lockstep fechou. O link MAVLink
                # de GCS (14550) ainda leva mais alguns segundos para publicar status
                # do estimador; conectar antes disso da heartbeat mas nenhuma mensagem
                # de health, e o voo falha com um timeout enganoso.
                print(f"[OK] PX4 conectado ao simulador; aquecendo {WARMUP_S:.0f}s ...")
                time.sleep(WARMUP_S)
                return proc
        except Exception:
            pass
        time.sleep(2.0)

    proc.kill()
    raise TimeoutError(
        f"PX4 nao conectou ao simulador em {timeout_s:.0f}s. "
        "O Unreal/Cosys-AirSim esta rodando? Veja o log em " + str(log_path)
    )


def px4_restart(log_path: Path, ip: str, vehicle: str,
                reset_airsim: bool = False) -> subprocess.Popen:
    """
    Derruba o PX4 e sobe de novo. DESLIGADO por padrao — ver aviso.

    ATENCAO: medido nesta bancada, reiniciar o PX4 com o lockstep ATIVO deixa o
    Cosys-AirSim preso a sessao TCP antiga. O novo PX4 imprime "Simulator
    connected", mas nao recebe dados de sensor: o EKF nunca converge e o voo falha
    com "sem mensagens de health", sintoma que nao aponta para a causa. O padrao
    observado era voo 1 ok (PX4 subiu com o AirSim recem-aberto), voo 2 em diante
    sempre falhando.

    Reiniciar o PX4 so e seguro reiniciando o Unreal junto, o que custa minutos por
    voo. Como ensure_gnss_enabled() e prepare_for_arm() ja tornam cada voo
    independente do anterior nos aspectos que afetam a medida, a campanha por
    padrao NAO reinicia.
    """
    print("\n[INFO] Reiniciando PX4 (--restart-px4) ...")
    px4_stop()
    if reset_airsim:
        airsim_reset(ip, vehicle)
    return px4_start(log_path)


def px4_ensure_running(log_path: Path) -> None:
    """Sobe o PX4 apenas se nao houver um rodando (modo padrao, sem restart)."""
    r = subprocess.run(
        ["wsl", "-e", "bash", "-lc", "pgrep -f px4_sitl_default/bin/px4"],
        capture_output=True, text=True, timeout=15,
    )
    if r.stdout.strip():
        return
    print("\n[INFO] Nenhum PX4 rodando; subindo ...")
    px4_start(log_path)


# ─────────────────────────────────────────────────────────────────────────────
# Execucao de um voo
# ─────────────────────────────────────────────────────────────────────────────

def flight_collided(out_dir: Path, run_name: str) -> bool:
    """Le do meta.json se o voo bateu na cena."""
    meta = out_dir / run_name / "meta.json"
    try:
        return bool(json.loads(meta.read_text(encoding="utf-8"))
                    .get("collision", {}).get("occurred"))
    except Exception:
        return False


def run_flight(run_name: str, out_dir: Path, deny_at_m: float, args,
               direction: int = 1) -> bool:
    """Roda um voo como subprocesso. Isolar o processo garante que uma falha de
    MAVSDK num voo nao contamine os seguintes."""
    cmd = [
        str(ROOT / "venv" / "Scripts" / "python.exe"), "-u",
        str(ROOT / "tools" / "run_gnss_denial_experiment.py"),
        "--run-name", run_name,
        "--out-dir", str(out_dir),
        "--deny-at-m", str(deny_at_m),
        "--direction", str(direction),
    ]
    if args.corridor_e is not None:
        cmd += ["--corridor-e", str(args.corridor_e)]
    cmd += [
        "--distance-m", str(args.distance_m),
        "--alt-m", str(args.alt_m),
        "--rate-hz", str(args.rate_hz),
        "--lidar-hz", str(args.lidar_hz),
        "--latency-s", str(args.latency_s),
        "--vehicle", args.vehicle,
        "--ip", args.ip,
        "--no-plot",
    ]
    print(f"[INFO] Voo: {run_name} (deny_at={deny_at_m} m)")
    r = subprocess.run(cmd, timeout=args.flight_timeout_s)
    ok = r.returncode == 0
    print(f"[{'OK' if ok else 'FALHA'}] {run_name}")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Agregacao
# ─────────────────────────────────────────────────────────────────────────────

def aggregate(out_dir: Path) -> dict:
    """Junta os runs da campanha e ajusta a lei de crescimento da deriva."""
    from analyze_gnss_campaign import aggregate_campaign
    return aggregate_campaign(out_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description="Campanha de repeticoes do experimento GNSS-denied")
    ap.add_argument("--reps", type=int, default=5, metavar="N",
                    help="Repeticoes com GNSS negado")
    ap.add_argument("--baseline-reps", type=int, default=2, metavar="N",
                    help="Repeticoes de referencia com GNSS ligado (piso de medicao)")
    ap.add_argument("--out-dir", type=Path, default=None, metavar="DIR")
    ap.add_argument("--deny-at-m", type=float, default=15.0, metavar="M")
    ap.add_argument("--distance-m", type=float, default=250.0, metavar="M")
    ap.add_argument("--alt-m", type=float, default=60.0, metavar="M")
    ap.add_argument("--rate-hz", type=float, default=20.0, metavar="HZ")
    ap.add_argument("--lidar-hz", type=float, default=4.0, metavar="HZ")
    ap.add_argument("--latency-s", type=float, default=0.255, metavar="S")
    ap.add_argument("--vehicle", default="PX4Drone", metavar="NOME")
    ap.add_argument("--ip", default="127.0.0.1", metavar="IP")
    ap.add_argument("--flight-timeout-s", type=float, default=600.0, metavar="S")
    ap.add_argument("--only-aggregate", action="store_true",
                    help="So agrega runs ja gravados em --out-dir")
    ap.add_argument("--corridor-e", type=float, default=None, metavar="M",
                    help="Coordenada E do corredor validado por probe_gnss_corridor.py")
    ap.add_argument("--max-retries", type=int, default=4, metavar="N",
                    help="Voos extras permitidos para repor runs que colidiram")
    ap.add_argument("--no-retry-on-collision", dest="retry_on_collision",
                    action="store_false",
                    help="Aceita runs com colisao em vez de repeti-los")
    ap.add_argument("--restart-px4", action="store_true",
                    help="Reinicia o PX4 entre voos. QUEBRA o lockstep do AirSim a partir "
                         "do 2o voo (ver px4_restart); so use reiniciando o Unreal junto.")
    ap.add_argument("--reset-airsim", action="store_true",
                    help="Reseta o simulador entre voos. QUEBRA o GPS do PX4 nesta "
                         "bancada (global_position_ok fica False); use por sua conta.")
    args = ap.parse_args()

    out_dir = args.out_dir or (
        Path("dataset_gnss_denial") / time.strftime("campaign_%Y%m%d_%H%M%S")
    )

    if args.only_aggregate:
        aggregate(out_dir)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    px4_log = out_dir / "px4_sitl.log"

    plan = [("baseline", -1.0, args.baseline_reps), ("denied", args.deny_at_m, args.reps)]
    results: list[dict] = []
    # Vaivem: alterna o sentido a cada voo para o veiculo repetir o mesmo trecho
    # de cidade, em vez de avancar para regiao nova (e nao validada) a cada run.
    direction = 1

    for kind, deny, n in plan:
        i = 0
        attempts = 0
        while i < n and attempts < n + args.max_retries:
            attempts += 1
            run_name = f"{kind}_rep{i + 1:02d}"
            try:
                if args.restart_px4:
                    px4_restart(px4_log, args.ip, args.vehicle, args.reset_airsim)
                else:
                    px4_ensure_running(px4_log)
                ok = run_flight(run_name, out_dir, deny, args, direction)
            except Exception as e:
                print(f"[ERRO] {run_name}: {e}")
                ok = False

            direction *= -1
            collided = flight_collided(out_dir, run_name)

            if ok and collided and args.retry_on_collision:
                # O trecho pre-colisao continua valido, mas para ter N repeticoes
                # comparaveis vale repetir. O voo fica gravado com sufixo _collided
                # em vez de descartado: e evidencia de que a cena tem obstaculos.
                bad = out_dir / f"{run_name}_collided_{attempts:02d}"
                try:
                    (out_dir / run_name).rename(bad)
                    print(f"[INFO] Voo colidiu; preservado como {bad.name} e repetido.")
                except Exception:
                    pass
                results.append({"run": bad.name, "kind": kind, "deny_at_m": deny,
                                "ok": True, "collided": True, "used": False})
                continue

            results.append({"run": run_name, "kind": kind, "deny_at_m": deny,
                            "ok": ok, "collided": collided, "used": ok})
            if ok:
                i += 1

    (out_dir / "campaign.json").write_text(
        json.dumps({"args": {k: str(v) for k, v in vars(args).items()}, "runs": results},
                   indent=2),
        encoding="utf-8",
    )
    n_ok = sum(1 for r in results if r["ok"])
    print(f"\n[CONCLUIDO] {n_ok}/{len(results)} voos ok -> {out_dir}")

    if n_ok:
        aggregate(out_dir)


if __name__ == "__main__":
    main()
