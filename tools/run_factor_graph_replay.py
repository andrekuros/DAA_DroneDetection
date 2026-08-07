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

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"

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


def optimize_fg(
    cols: dict[str, np.ndarray],
    key_hz: float = 2.0,
    vo_sigma_m: float = 0.12,
    imu_sigma_m: float = 0.8,
    baro_sigma_m: float = 1.5,
    prior_sigma_m: float = 0.05,
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
    n = len(keys)

    gt = np.column_stack([cols["cosys_x"], cols["cosys_y"], cols["cosys_z"]])
    px4 = np.column_stack([cols["px4_x"], cols["px4_y"], cols["px4_z"]])
    # alinhar PX4 ao GT no instante da negacao (mesmo offset do orquestrador)
    offset = gt[0] - px4[0]
    px4_aln = px4 + offset

    baro = cols["baro_alt_m"]
    # baro nesta cena nasce ~120 m acima do NED alt; usamos so o DELTA desde a negacao
    baro0 = baro[0]
    alt0 = -gt[0, 2]

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

            # VO relativo simulado (forte): verdade + ruido deterministico por seed
            rng = np.random.default_rng(1000 + k)
            dp_vo = (gt[i1] - gt[i0]) + rng.normal(0.0, vo_sigma_m, size=3)
            r_vo = (P[k] - P[k - 1] - dp_vo) / max(vo_sigma_m, 1e-3)
            res.extend(r_vo.tolist())

            # baro: altitude relativa desde a negacao
            d_alt_baro = float(baro[i1] - baro0)
            d_alt_state = float(-(P[k, 2] - P[0, 2]))
            res.append((d_alt_state - d_alt_baro) / baro_sigma_m)

        return np.asarray(res, dtype=float)

    if not HAS_SCIPY:
        raise RuntimeError("scipy e necessario (pip install scipy)")

    sol = least_squares(residuals, x0, method="trf", max_nfev=200, verbose=0)
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
        "vo_sigma_m": vo_sigma_m,
    }
    return t_keys, P, {"gt": gt[keys], "px4": px4_aln[keys], "err_fg": err,
                       "err_ekf": err_ekf, "meta": meta}


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


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay factor-graph nas janelas GNSS-denied")
    ap.add_argument("campaign_dir", type=Path)
    ap.add_argument("--prefix", default="gnss_wsc250_fg", help="Prefixo em figures/")
    ap.add_argument("--vo-sigma-m", type=float, default=0.12, metavar="M",
                    help="Ruido dos fatores VO simulados (m)")
    ap.add_argument("--key-hz", type=float, default=2.0)
    args = ap.parse_args()

    runs_out = []
    summaries = []
    for rd in sorted(args.campaign_dir.iterdir()):
        if not rd.is_dir() or "denied_rep" not in rd.name or "collided" in rd.name:
            continue
        csv_path = rd / "telemetry.csv"
        if not csv_path.is_file():
            continue
        print(f"[INFO] {rd.name} ...")
        cols = _denied_slice(_load_run(csv_path))
        t, P, pack = optimize_fg(
            cols, key_hz=args.key_hz, vo_sigma_m=args.vo_sigma_m,
        )
        pack["meta"]["run"] = rd.name
        print(f"       EKF final={pack['meta']['err_ekf_final_m']:.1f} m  "
              f"FG final={pack['meta']['err_fg_final_m']:.1f} m  "
              f"x{pack['meta']['improvement_final_x']:.1f}")
        runs_out.append({
            "name": rd.name, "t": t, "fg": P, "gt": pack["gt"], "px4": pack["px4"],
            "err_fg": pack["err_fg"], "err_ekf": pack["err_ekf"], "meta": pack["meta"],
        })
        summaries.append(pack["meta"])

    if not runs_out:
        raise SystemExit(f"Nenhum denied_rep* em {args.campaign_dir}")

    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_comparison(runs_out, FIGURES / args.prefix)

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
            "Visual relative-pose factors are simulated from ground truth + noise "
            "(no camera log in this campaign). IMU and baro factors use the CSV."
        ),
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
