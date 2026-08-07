"""
analyze_gnss_campaign.py — Lei de deriva e barras de erro da campanha
=====================================================================
Agrega os voos de uma campanha (run_gnss_campaign.py) e produz o que sustenta o
extended abstract:

  1. Lei de crescimento da deriva: ajusta |e|(t) = A * t^p sobre a janela sem
     GNSS, reportando o expoente p. Dead-reckoning inercial puro com bias de
     acelerometro preve p ~ 2; o valor medido e o numero que um estimador com
     factor graph tera que bater.
  2. Envelope mediano +/- percentis sobre as repeticoes (barras de erro).
  3. Piso de medicao a partir dos voos de referencia com GNSS ligado.
  4. Estrutura visivel ao LiDAR durante a janela sem GNSS — evidencia de que ha
     observabilidade para fatores de scan-matching justamente onde o GNSS falha.

Uso:
    python tools/analyze_gnss_campaign.py dataset_gnss_denial/campaign_20260807_010203
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"

COLOR_DENIED = "#E74C3C"
COLOR_BASELINE = "#2ECC71"
COLOR_BAND = "#E74C3C"
COLOR_LIDAR = "#3498DB"


def _load(csv_path: Path) -> dict[str, list[float]]:
    cols: dict[str, list[float]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                try:
                    val = float(v)
                except (TypeError, ValueError):
                    val = float("nan")
                cols.setdefault(k, []).append(val)
    return cols


def _fit_power_law(ts: list[float], es: list[float]) -> tuple[float, float, float]:
    """
    Ajusta |e| = A * t^p por regressao linear em log-log.

    Retorna (A, p, r2). Pontos com t<=0 ou e<=0 sao descartados: log nao existe
    la, e o inicio da janela e dominado pelo residuo do EKF, nao pela deriva.
    """
    xs, ys = [], []
    for t, e in zip(ts, es):
        if t > 0.2 and e > 1e-6 and math.isfinite(t) and math.isfinite(e):
            xs.append(math.log(t))
            ys.append(math.log(e))
    n = len(xs)
    if n < 5:
        return float("nan"), float("nan"), float("nan")

    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return float("nan"), float("nan"), float("nan")
    p = sxy / sxx
    lnA = my - p * mx
    ss_res = sum((y - (lnA + p * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
    return math.exp(lnA), p, r2


def _percentile(vals: list[float], q: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    k = (len(s) - 1) * q
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return s[lo] if lo == hi else s[lo] + (k - lo) * (s[hi] - s[lo])


def _resample(ts: list[float], es: list[float], grid: list[float]) -> list[float]:
    """Reamostra uma serie numa grade comum, para empilhar repeticoes."""
    out = []
    for g in grid:
        best_i, best_d = None, float("inf")
        for i, t in enumerate(ts):
            d = abs(t - g)
            if d < best_d:
                best_d, best_i = d, i
        out.append(es[best_i] if best_i is not None and best_d < 0.5 else float("nan"))
    return out


def aggregate_campaign(out_dir: Path, prefix: str | None = None) -> dict:
    """Agrega os runs, ajusta a lei de deriva e gera as figuras da campanha."""
    runs = sorted(p for p in out_dir.iterdir() if p.is_dir() and (p / "telemetry.csv").is_file())
    if not runs:
        raise ValueError(f"Nenhum run com telemetry.csv em {out_dir}")

    denied, baseline, fits, lidar_pts = [], [], [], []

    for rd in runs:
        cols = _load(rd / "telemetry.csv")
        if "t_since_denial" not in cols or "err_norm" not in cols:
            print(f"[SKIP] {rd.name}: CSV sem as colunas esperadas (run antigo?)")
            continue

        td = cols["t_since_denial"]
        err = cols["err_norm"]
        pairs = [(t, e) for t, e in zip(td, err) if t >= 0]

        if pairs:
            ts, es = zip(*pairs)
            denied.append((rd.name, list(ts), list(es)))
            A, p, r2 = _fit_power_law(list(ts), list(es))
            fits.append({"run": rd.name, "A": A, "p": p, "r2": r2,
                         "err_final_m": es[-1], "exposure_s": ts[-1]})
            # Estrutura vista pelo LiDAR durante a janela sem GNSS
            if "lidar_points" in cols:
                pts = [n for t, n in zip(td, cols["lidar_points"]) if t >= 0 and n >= 0]
                if pts:
                    lidar_pts.extend(pts)
        else:
            baseline.append((rd.name, cols["t_mono"], err))

    summary: dict = {
        "campaign": out_dir.name,
        "n_denied": len(denied),
        "n_baseline": len(baseline),
        "fits": fits,
    }

    if fits:
        ps = [f["p"] for f in fits if math.isfinite(f["p"])]
        finals = [f["err_final_m"] for f in fits]
        summary["exponent_p_median"] = _percentile(ps, 0.5)
        summary["exponent_p_min"] = min(ps) if ps else None
        summary["exponent_p_max"] = max(ps) if ps else None
        summary["err_final_median_m"] = _percentile(finals, 0.5)
        summary["err_final_min_m"] = min(finals)
        summary["err_final_max_m"] = max(finals)

    if baseline:
        allb = [e for _, _, es in baseline for e in es if math.isfinite(e)]
        summary["baseline_floor_median_m"] = _percentile(allb, 0.5)
        summary["baseline_floor_p95_m"] = _percentile(allb, 0.95)

    if lidar_pts:
        summary["lidar_points_median"] = _percentile(lidar_pts, 0.5)
        summary["lidar_points_p05"] = _percentile(lidar_pts, 0.05)
        summary["frac_frames_with_structure"] = sum(1 for n in lidar_pts if n > 0) / len(lidar_pts)

    slug = prefix or f"gnss_{out_dir.name}"
    FIGURES.mkdir(parents=True, exist_ok=True)
    if denied:
        _plot_drift_law(denied, baseline, summary, FIGURES / f"{slug}_drift_law")
    if lidar_pts:
        _plot_lidar_structure(denied, out_dir, FIGURES / f"{slug}_lidar_structure")

    out_json = out_dir / "campaign_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (FIGURES / f"{slug}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n[RESUMO] {out_dir.name}")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:32s} = {v:.3f}")
        elif isinstance(v, int):
            print(f"  {k:32s} = {v}")
    print(f"  -> {out_json}")
    return summary


def _plot_drift_law(denied, baseline, summary, outfile: Path) -> None:
    """Envelope da deriva sobre as repeticoes, com a lei ajustada."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

    t_max = min(max(ts) for _, ts, _ in denied)
    grid = [i * 0.5 for i in range(int(t_max / 0.5) + 1)]
    stacked = [_resample(ts, es, grid) for _, ts, es in denied]

    med, lo, hi = [], [], []
    for i in range(len(grid)):
        vals = [s[i] for s in stacked if math.isfinite(s[i])]
        med.append(_percentile(vals, 0.5))
        lo.append(_percentile(vals, 0.05))
        hi.append(_percentile(vals, 0.95))

    ax.fill_between(grid, lo, hi, color=COLOR_BAND, alpha=0.18,
                    label=f"p05-p95 ({len(denied)} voos)")
    ax.plot(grid, med, color=COLOR_DENIED, linewidth=2.5, label="Mediana - GNSS negado")

    p = summary.get("exponent_p_median")
    if p and math.isfinite(p):
        A = _percentile([f["A"] for f in summary["fits"] if math.isfinite(f["A"])], 0.5)
        fit = [A * (g ** p) if g > 0 else float("nan") for g in grid]
        ax.plot(grid, fit, color="#34495E", linestyle="--", linewidth=1.6,
                label=f"Ajuste $|e| = {A:.3f}\\,t^{{{p:.2f}}}$")

    floor = summary.get("baseline_floor_p95_m")
    if floor and math.isfinite(floor):
        ax.axhline(floor, color=COLOR_BASELINE, linestyle=":", linewidth=1.8)
        ax.annotate(f"Piso de medicao (GNSS ligado, p95 = {floor:.2f} m)",
                    (grid[len(grid) // 2], floor), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=9, color="#27AE60")

    ax.set_xlabel("Tempo desde a negacao de GNSS (s)", fontsize=11)
    ax.set_ylabel("Erro de posicao |e| (m)", fontsize=11)
    ax.set_title("Lei de crescimento da deriva inercial sem GNSS (PX4 EKF2)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.35)

    fig.tight_layout()
    fig.savefig(outfile.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(outfile.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {outfile.stem}")


def _plot_lidar_structure(denied, out_dir: Path, outfile: Path) -> None:
    """
    Estrutura vista pelo LiDAR durante a janela sem GNSS.

    Serve ao argumento central do trabalho futuro: o ambiente que nega GNSS
    (canion urbano) e o mesmo que oferece estrutura para fatores de scan-matching.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=150)
    plotted = False
    for name, *_ in denied:
        cols = _load(out_dir / name / "telemetry.csv")
        if "lidar_points" not in cols:
            continue
        pairs = [(t, n) for t, n in zip(cols["t_since_denial"], cols["lidar_points"])
                 if t >= 0 and n >= 0]
        if not pairs:
            continue
        ts, ns = zip(*pairs)
        ax.plot(ts, ns, color=COLOR_LIDAR, alpha=0.55, linewidth=1.2)
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_xlabel("Tempo desde a negacao de GNSS (s)", fontsize=11)
    ax.set_ylabel("Pontos LiDAR por varredura", fontsize=11)
    ax.set_title("Estrutura disponivel para scan-matching durante a janela sem GNSS",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.35)

    fig.tight_layout()
    fig.savefig(outfile.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(outfile.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {outfile.stem}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Agrega uma campanha GNSS-denied")
    ap.add_argument("campaign_dir", type=Path, help="Pasta da campanha")
    ap.add_argument("--prefix", default=None, help="Prefixo dos arquivos em figures/")
    args = ap.parse_args()

    if not args.campaign_dir.is_dir():
        ap.error(f"nao encontrado: {args.campaign_dir}")
    aggregate_campaign(args.campaign_dir, args.prefix)


if __name__ == "__main__":
    main()
