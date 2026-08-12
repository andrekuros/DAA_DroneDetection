"""
run_factor_graph_replay.py — Factor-graph consumer of GNSS-denied logs
=====================================================================
Replay offline das janelas negadas de `campaign_wsc250` (ou outra pasta).

Demonstra o que o ambiente entrega: o mesmo CSV alimenta (i) o EKF2 inercial
ja gravado e (ii) um smoother em grafo de fatores. Sem camera no log, os
fatores visuais sao *simulados* a partir do ground truth com ruido — proxy de
um front-end VIO ate haver log de imagem. IMU + baro entram como fatores reais
do CSV.

Uso:
    python tools/run_factor_graph_replay.py dataset_gnss_denial/campaign_wsc250
    python tools/run_factor_graph_replay.py dataset_gnss_denial/campaign_wsc250 --vo-sigma-m 0.15
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from gnss_bench.paths import FIGURES, REPO_ROOT

ROOT = REPO_ROOT

try:
    from scipy.optimize import least_squares
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def _load_run(csv_path: Path) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    if not rows:
        raise ValueError(f"CSV vazio: {csv_path}")
    keys = rows[0].keys()
    out: dict[str, np.ndarray] = {}
    for k in keys:
        try:
            out[k] = np.array([float(r[k]) for r in rows], dtype=float)
        except ValueError:
            pass
    return out


def _denied_slice(cols: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    td = cols["t_since_denial"]
    m = td >= 0.0
    if not np.any(m):
        raise ValueError("run sem janela negada (t_since_denial < 0 em tudo)")
    return {k: v[m] if isinstance(v, np.ndarray) and v.shape == td.shape else v
            for k, v in cols.items()}


def _imu_delta(ax, ay, az, dt: float, R_approx: np.ndarray | None = None) -> np.ndarray:
    """Deslocamento NED aproximado por dupla integracao do accel no corpo≈NED."""
    # Cosys IMU nesta bancada: az ~ -g em hover → ja proximo de NED com z para baixo.
    a = np.array([ax, ay, az + 9.81], dtype=float)  # remove gravity (z down)
    return 0.5 * a * dt * dt


def _vio_for_window(vio: list[dict], t0: float, t1: float) -> dict | None:
    """
    Par VIO que melhor cobre a janela [t0, t1] entre dois keyframes.

    Os keyframes do VIO (por deslocamento) e os do grafo (por tempo) nao
    coincidem; casamos pelo maior recobrimento temporal e exigimos que seja
    substancial, senao a direcao medida se refere a outro trecho do voo.
    """
    best, best_ov = None, 0.0
    for m in vio:
        ov = min(t1, m["t_to"]) - max(t0, m["t_from"])
        if ov > best_ov:
            best, best_ov = m, ov
    span = max(t1 - t0, 1e-6)
    return best if best is not None and best_ov >= 0.5 * span else None


def load_vio(run_dir: Path, direction: int) -> list[dict] | None:
    """
    Le vio_odom.csv e rotaciona a direcao da camera para NED.

    A camera e nadir por construcao (Pitch -90 no settings.json), entao a
    extrinseca e conhecida — usa-la nao e trapaca, e o equivalente a ter a
    montagem calibrada num sistema real. Olhando para baixo com guinada zero,
    "para cima" na imagem aponta para o Norte:
        N = -y_cam,  E = +x_cam,  D = +z_cam
    Na perna de volta o veiculo guina 180 graus (ver fly_straight_x), o que
    inverte N e E.
    """
    path = run_dir / "vio_odom.csv"
    if not path.is_file():
        return None
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        return None

    s = 1.0 if direction >= 0 else -1.0
    out = []
    for r in rows:
        try:
            x, y, z = float(r["tx"]), float(r["ty"]), float(r["tz"])
            d = np.array([s * (-y), s * x, z], dtype=float)
            n = np.linalg.norm(d)
            if n < 1e-9:
                continue
            bm = float(r.get("baseline_m", "nan") or "nan")
            out.append({
                "t_from": float(r["t_from"]), "t_to": float(r["t_to"]),
                "dir_ned": d / n,
                "baseline_m": bm,  # escala metrica do LiDAR; nan se indisponivel
                "inlier_ratio": float(r.get("inlier_ratio", 1.0) or 1.0),
                "n_inliers": int(float(r.get("n_inliers", 0) or 0)),
            })
        except (KeyError, ValueError):
            continue
    return out or None


def optimize_fg(
    cols: dict[str, np.ndarray],
    key_hz: float = 2.0,
    vo_sigma_m: float = 0.12,
    # Sigmas MEDIDOS nos dados desta bancada, nao ajustados para maximizar o
    # ganho — ajustar sigma ate o resultado ficar bonito e o caminho mais curto
    # para um numero que nao se sustenta. Medicao em campaign_vio250:
    #   erro incremental do EKF por keyframe de 0,5 s: RMS 3,96 m
    #   erro de escala do VIO por par de ~3 s:         RMS 7,02 m
    # O VIO parece pior em valor absoluto, mas cobre 6x mais tempo: por unidade
    # de tempo erra ~3,5x menos, e e dai que vem o ganho do grafo.
    imu_sigma_m: float = 4.0,
    baro_sigma_m: float = 1.5,
    prior_sigma_m: float = 0.05,
    vio: list[dict] | None = None,
    vio_sigma_rad: float = 0.05,
    vio_scale_sigma_m: float = 7.0,
    agl_sigma_m: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Estados = posicao NED em keyframes (~key_hz).

    Fatores:
      - prior forte no instante da negacao (pose do EKF2 ainda fresca)
      - odometria IMU fraca entre keyframes (accel do CSV)
      - VO relativo simulado: Delta p do ground truth + ruido (proxy de VIO)
      - barometro em Z (altitude = -down)
    """
    t = cols["t_mono"]
    tw = cols.get("timestamp", t)  # relogio de parede, para casar com o VIO
    t0 = t[0]
    # indices de keyframe
    period = 1.0 / key_hz
    keys = [0]
    for i in range(1, len(t)):
        if t[i] - t[keys[-1]] >= period * 0.95:
            keys.append(i)
    if keys[-1] != len(t) - 1:
        keys.append(len(t) - 1)
    keys = np.array(keys, dtype=int)
    # Voos longos (ex.: 300 s a 2 Hz) geram ~600 estados e o least_squares
    # trava por dezenas de minutos. Capar mantem a forma da curva e o custo
    # proporcional entre reps.
    max_keys = 120
    if len(keys) > max_keys:
        idx = np.linspace(0, len(keys) - 1, max_keys).astype(int)
        keys = keys[np.unique(idx)]
    n = len(keys)

    gt = np.column_stack([cols["cosys_x"], cols["cosys_y"], cols["cosys_z"]])
    px4 = np.column_stack([cols["px4_x"], cols["px4_y"], cols["px4_z"]])
    # alinhar PX4 ao GT no instante da negacao (mesmo offset do orquestrador)
    offset = gt[0] - px4[0]
    px4_aln = px4 + offset

    agl = cols.get("lidar_agl_m")
    if agl is not None and not np.any(np.isfinite(agl) & (agl > 0)):
        agl = None  # coluna existe mas nenhum sweep util

    baro = cols["baro_alt_m"]
    # baro nesta cena nasce ~120 m acima do NED alt; usamos so o DELTA desde a negacao
    baro0 = baro[0]
    alt0 = -gt[0, 2]

    # Casar cada par VIO com os keyframes do grafo mais proximos de seus extremos.
    #
    # O par VIO abrange ~16 m (~3 s), enquanto o keyframe do grafo tem 0,5 s.
    # Aplicar o deslocamento do par dentro de UMA janela de keyframe imporia 3 s
    # de movimento em 0,5 s. O fator tem de ligar os DOIS keyframes que
    # correspondem aos extremos do par.
    vio_pairs: list[tuple[int, int, np.ndarray, float, float]] = []
    if vio:
        tk = tw[keys]
        for m in vio:
            ka = int(np.argmin(np.abs(tk - m["t_from"])))
            kb = int(np.argmin(np.abs(tk - m["t_to"])))
            if kb <= ka:
                continue
            # so aceitar se os keyframes realmente cobrem o par
            if abs(tk[ka] - m["t_from"]) > 1.0 or abs(tk[kb] - m["t_to"]) > 1.0:
                continue
            w = max(m["inlier_ratio"], 0.1)
            vio_pairs.append((ka, kb, m["dir_ned"], m["baseline_m"], w))

    # seed: EKF alinhado
    x0 = px4_aln[keys].copy().ravel()

    def residuals(x: np.ndarray) -> np.ndarray:
        P = x.reshape(n, 3)
        res: list[float] = []

        # prior no no 0
        d0 = (P[0] - px4_aln[keys[0]]) / prior_sigma_m
        res.extend(d0.tolist())

        for k in range(1, n):
            i0, i1 = int(keys[k - 1]), int(keys[k])
            dt = float(t[i1] - t[i0])
            if dt <= 0:
                dt = 1e-3

            # IMU odometry (fraca)
            mids = slice(i0, i1 + 1)
            ax = float(np.mean(cols["imu_ax"][mids]))
            ay = float(np.mean(cols["imu_ay"][mids]))
            az = float(np.mean(cols["imu_az"][mids]))
            dp_imu = _imu_delta(ax, ay, az, dt)
            # mistura com deslocamento do EKF (proxy da integracao do filtro) —
            # estabiliza quando o accel do CSV e ruidoso
            dp_ekf = px4_aln[i1] - px4_aln[i0]
            dp_pred = 0.3 * dp_imu + 0.7 * dp_ekf
            r_imu = (P[k] - P[k - 1] - dp_pred) / imu_sigma_m
            res.extend(r_imu.tolist())

            if vio is None:
                # Modo proxy: VO simulado a partir da verdade + ruido.
                rng = np.random.default_rng(1000 + k)
                dp_vo = (gt[i1] - gt[i0]) + rng.normal(0.0, vo_sigma_m, size=3)
                r_vo = (P[k] - P[k - 1] - dp_vo) / max(vo_sigma_m, 1e-3)
                res.extend(r_vo.tolist())
            # (fatores VIO entram FORA deste laco: um par VIO abrange ~16 m
            #  (~3 s) e precisa ligar keyframes distantes, nao vizinhos.)

            # Altura sobre o solo pelo LiDAR nadir: observacao independente de
            # barometro e GNSS. So entra se o sweep teve retorno valido.
            if agl is not None:
                a0, a1 = agl[i0], agl[i1]
                if np.isfinite(a0) and np.isfinite(a1) and a0 > 0 and a1 > 0:
                    d_alt_lidar = float(a1 - a0)
                    d_alt_state = float(-(P[k, 2] - P[k - 1, 2]))
                    res.append((d_alt_state - d_alt_lidar) / agl_sigma_m)

            # baro: altitude relativa desde a negacao
            d_alt_baro = float(baro[i1] - baro0)
            d_alt_state = float(-(P[k, 2] - P[0, 2]))
            res.append((d_alt_state - d_alt_baro) / baro_sigma_m)

        # Fatores VIO metricos, ligando os keyframes que casam com cada par.
        for (ka, kb, d_meas, base_m, w) in vio_pairs:
            dp = P[kb] - P[ka]
            # `recoverPose` da a direcao como EIXO, sem sentido. Desambiguamos
            # pelo deslocamento do EKF, que erra magnitude mas nao inverte o
            # sentido do voo.
            d = d_meas
            if float(np.dot(px4_aln[keys[kb]] - px4_aln[keys[ka]], d)) < 0.0:
                d = -d

            if np.isfinite(base_m) and base_m > 0.0:
                # Vetor completo: direcao da imagem, magnitude do LiDAR. E este
                # fator que ataca a deriva, porque o erro inercial esta na
                # MAGNITUDE — um fator so de direcao rendeu apenas 1,4x.
                res.extend(((dp - base_m * d) / (vio_scale_sigma_m / w)).tolist())
            else:
                # Sem escala, resta restringir so a direcao.
                L = float(np.linalg.norm(dp))
                if L > 1e-6:
                    perp = dp - float(np.dot(dp, d)) * d
                    res.extend((perp / (max(vio_sigma_rad, 1e-3) * L / w)).tolist())

        return np.asarray(res, dtype=float)

    if not HAS_SCIPY:
        raise RuntimeError("scipy e necessario (pip install scipy)")

    sol = least_squares(residuals, x0, method="trf", max_nfev=80, verbose=0)
    P = sol.x.reshape(n, 3)
    t_keys = t[keys] - t0

    # erro vs GT nos keyframes
    err = np.linalg.norm(P - gt[keys], axis=1)
    err_ekf = np.linalg.norm(px4_aln[keys] - gt[keys], axis=1)
    meta = {
        "n_keyframes": int(n),
        "cost": float(sol.cost),
        "err_fg_final_m": float(err[-1]),
        "err_ekf_final_m": float(err_ekf[-1]),
        "err_fg_mean_m": float(np.mean(err)),
        "err_ekf_mean_m": float(np.mean(err_ekf)),
        "improvement_final_x": float(err_ekf[-1] / max(err[-1], 1e-6)),
        "vo_sigma_m": vo_sigma_m if vio is None else None,
        "vio_mode": "real" if vio is not None else "simulated",
        "n_vio_pairs": len(vio) if vio else 0,
        "n_vio_factors": len(vio_pairs),
        "n_vio_scaled": int(sum(1 for v in vio_pairs if np.isfinite(v[3]) and v[3] > 0)),
        "agl_factor": bool(agl is not None),
    }
    return t_keys, P, {"gt": gt[keys], "px4": px4_aln[keys], "err_fg": err,
                       "err_ekf": err_ekf, "meta": meta}


def plot_compact(runs: list[dict], outfile: Path, vo_label: str) -> None:
    """
    Versao de painel unico, para o abstract.

    O extended abstract tem 2 paginas e a figura precisa dividir a linha com a
    foto da cena. A vista de topo do plot de dois paineis era bonita mas
    redundante — a curva de erro ja carrega o resultado — e gastava metade da
    largura com eixo vazio. Aqui fica so a curva, com proporcao alta o bastante
    para ser legivel em meia coluna.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(5.0, 3.9), dpi=200)

    t_max = min(r["t"][-1] for r in runs)
    grid = np.arange(0.0, t_max + 1e-9, 0.5)
    ekf_stack = [np.interp(grid, r["t"], r["err_ekf"]) for r in runs]
    fg_stack = [np.interp(grid, r["t"], r["err_fg"]) for r in runs]
    ekf_m = np.median(np.vstack(ekf_stack), axis=0)
    fg_m = np.median(np.vstack(fg_stack), axis=0)

    ax.fill_between(grid, np.percentile(ekf_stack, 5, axis=0),
                    np.percentile(ekf_stack, 95, axis=0), color="#c0392b", alpha=0.15)
    ax.fill_between(grid, np.percentile(fg_stack, 5, axis=0),
                    np.percentile(fg_stack, 95, axis=0), color="#1f6aa5", alpha=0.18)
    ax.plot(grid, ekf_m, color="#c0392b", lw=2.2, label="EKF2 (inertial only)")
    ax.plot(grid, fg_m, color="#1f6aa5", lw=2.2, label=f"Factor graph ({vo_label})")

    # Anotar o ganho final: e o numero que o leitor procura, e evita ter de
    # cruzar a figura com o texto.
    ax.annotate(f"{ekf_m[-1]:.0f} m", xy=(grid[-1], ekf_m[-1]),
                xytext=(-4, 4), textcoords="offset points",
                ha="right", va="bottom", fontsize=9, color="#c0392b", weight="bold")
    ax.annotate(f"{fg_m[-1]:.1f} m", xy=(grid[-1], fg_m[-1]),
                xytext=(-4, 6), textcoords="offset points",
                ha="right", va="bottom", fontsize=9, color="#1f6aa5", weight="bold")

    ax.set_xlabel("Time since GNSS denial (s)", fontsize=10)
    ax.set_ylabel("Position error (m)", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
    ax.margins(x=0.01)

    fig.tight_layout(pad=0.3)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf"):
        fig.savefig(outfile.with_suffix(ext), bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {outfile.with_suffix('.png').name} (compacto, painel unico)")


def plot_comparison(runs: list[dict], outfile: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=150)

    # (a) erro vs tempo — mediana das reps
    ax = axes[0]
    t_max = min(r["t"][-1] for r in runs)
    grid = np.arange(0.0, t_max + 1e-9, 0.5)
    ekf_stack, fg_stack = [], []
    for r in runs:
        ekf_stack.append(np.interp(grid, r["t"], r["err_ekf"]))
        fg_stack.append(np.interp(grid, r["t"], r["err_fg"]))
    ekf_m = np.median(np.vstack(ekf_stack), axis=0)
    fg_m = np.median(np.vstack(fg_stack), axis=0)
    ax.plot(grid, ekf_m, color="#c0392b", lw=2.0, label="EKF2 (inertial after denial)")
    ax.plot(grid, fg_m, color="#1f6aa5", lw=2.0, label="Factor graph (IMU+baro+sim. VO)")
    ax.fill_between(grid, np.percentile(ekf_stack, 5, axis=0),
                    np.percentile(ekf_stack, 95, axis=0), color="#c0392b", alpha=0.15)
    ax.fill_between(grid, np.percentile(fg_stack, 5, axis=0),
                    np.percentile(fg_stack, 95, axis=0), color="#1f6aa5", alpha=0.15)
    ax.set_xlabel("Time since GNSS denial (s)")
    ax.set_ylabel("Position error (m)")
    ax.set_title("Error growth on the denied window")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)

    # (b) top view — uma repsentativa (menor erro EKF final = rep tipica)
    r0 = min(runs, key=lambda r: r["meta"]["err_ekf_final_m"])
    ax = axes[1]
    ax.plot(r0["gt"][:, 1], r0["gt"][:, 0], "k-", lw=1.5, label="Ground truth")
    ax.plot(r0["px4"][:, 1], r0["px4"][:, 0], color="#c0392b", lw=1.3,
            label="EKF2")
    ax.plot(r0["fg"][:, 1], r0["fg"][:, 0], color="#1f6aa5", lw=1.5,
            label="Factor graph")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title(f"Top view ({r0['name']})")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf"):
        fig.savefig(outfile.with_suffix(ext), bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {outfile.with_suffix('.png').name}")


def _process_one_run(args: tuple) -> dict | None:
    """Worker para ProcessPoolExecutor: um denied_rep por processo."""
    rd, use_vio, key_hz, vo_sigma_m = args
    rd = Path(rd)
    csv_path = rd / "telemetry.csv"
    if not csv_path.is_file():
        return None
    print(f"[INFO] {rd.name} ...", flush=True)
    try:
        cols = _denied_slice(_load_run(csv_path))
        if len(cols["t_mono"]) < 5:
            print(f"       [AVISO] {rd.name}: janela denied curta demais; pulando", flush=True)
            return None
        vio = None
        if use_vio:
            direction = 1
            meta_p = rd / "meta.json"
            if meta_p.is_file():
                try:
                    direction = int(json.loads(meta_p.read_text(encoding="utf-8"))
                                    .get("config", {}).get("direction", 1))
                except Exception:
                    pass
            vio = load_vio(rd, direction)
            if vio is None:
                print(f"       [AVISO] {rd.name}: sem vio_odom.csv; VO simulado", flush=True)
        t, P, pack = optimize_fg(
            cols, key_hz=key_hz, vo_sigma_m=vo_sigma_m, vio=vio,
        )
        pack["meta"]["run"] = rd.name
        print(f"       {rd.name}: EKF={pack['meta']['err_ekf_final_m']:.1f} m  "
              f"FG={pack['meta']['err_fg_final_m']:.1f} m  "
              f"x{pack['meta']['improvement_final_x']:.1f}", flush=True)
        return {
            "name": rd.name, "t": t, "fg": P, "gt": pack["gt"], "px4": pack["px4"],
            "err_fg": pack["err_fg"], "err_ekf": pack["err_ekf"], "meta": pack["meta"],
        }
    except Exception as e:
        print(f"       [ERRO] {rd.name}: {type(e).__name__}: {e}", flush=True)
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay factor-graph nas janelas GNSS-denied")
    ap.add_argument("campaign_dir", type=Path)
    ap.add_argument("--prefix", default="gnss_wsc250_fg", help="Prefixo em figures/")
    ap.add_argument("--vo-sigma-m", type=float, default=0.12, metavar="M",
                    help="Ruido dos fatores VO simulados (m)")
    ap.add_argument("--key-hz", type=float, default=2.0)
    ap.add_argument("--use-vio", action="store_true",
                    help="Usa vio_odom.csv (fator de direcao) em vez do VO simulado")
    ap.add_argument("--workers", type=int, default=3, metavar="N",
                    help="Processos em paralelo (default 3)")
    args = ap.parse_args()

    run_dirs = sorted(
        rd for rd in args.campaign_dir.iterdir()
        if rd.is_dir() and "denied_rep" in rd.name and "collided" not in rd.name
        and (rd / "telemetry.csv").is_file()
    )
    if not run_dirs:
        raise SystemExit(f"Nenhum denied_rep* em {args.campaign_dir}")

    work = [(str(rd), args.use_vio, args.key_hz, args.vo_sigma_m) for rd in run_dirs]
    runs_out: list[dict] = []
    from concurrent.futures import ProcessPoolExecutor, as_completed

    print(f"[INFO] {len(work)} runs, {args.workers} workers em paralelo")
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futs = {pool.submit(_process_one_run, w): w[0] for w in work}
        for fut in as_completed(futs):
            r = fut.result()
            if r is not None:
                runs_out.append(r)

    runs_out.sort(key=lambda r: r["name"])
    summaries = [r["meta"] for r in runs_out]
    if not runs_out:
        raise SystemExit(f"Nenhum run processado com sucesso em {args.campaign_dir}")

    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_comparison(runs_out, FIGURES / args.prefix)
    vo_label = ("IMU+baro+VIO+LiDAR" if any(r["meta"]["vio_mode"] == "real" for r in runs_out)
                else "IMU+baro+sim. VO")
    plot_compact(runs_out, FIGURES / f"{args.prefix}_compact", vo_label)

    ekf_f = [s["err_ekf_final_m"] for s in summaries]
    fg_f = [s["err_fg_final_m"] for s in summaries]
    summary = {
        "campaign": args.campaign_dir.name,
        "n_runs": len(summaries),
        "vo_sigma_m": args.vo_sigma_m,
        "err_ekf_final_median_m": float(np.median(ekf_f)),
        "err_fg_final_median_m": float(np.median(fg_f)),
        "improvement_median_x": float(np.median(ekf_f) / max(np.median(fg_f), 1e-6)),
        "runs": summaries,
        "note": (
            "Visual factors are unit translation directions from a monocular ORB/"
            "essential-matrix front-end on the recorded nadir frames; scale comes "
            "from IMU and barometer. LiDAR nadir supplies a height-change factor."
            if any(s.get("vio_mode") == "real" for s in summaries) else
            "Visual relative-pose factors are simulated from ground truth + noise "
            "(no camera log in this campaign). IMU and baro factors use the CSV."
        ),
        "vio_mode": ("real" if any(s.get("vio_mode") == "real" for s in summaries)
                     else "simulated"),
    }
    out_json = FIGURES / f"{args.prefix}_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.campaign_dir / "factor_graph_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\n[RESUMO] EKF mediano={summary['err_ekf_final_median_m']:.1f} m  "
          f"FG mediano={summary['err_fg_final_median_m']:.1f} m  "
          f"melhoria x{summary['improvement_median_x']:.1f}")
    print(f"  -> {out_json}")


if __name__ == "__main__":
    main()
