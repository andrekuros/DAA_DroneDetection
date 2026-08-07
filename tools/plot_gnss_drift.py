"""
plot_gnss_drift.py — Figuras do experimento GNSS-denied
========================================================
Le o telemetry.csv produzido por run_gnss_denial_experiment.py e gera as figuras
de deriva para o artigo: trajetoria 3D, vista superior XY e erro vs. tempo.

Segue as convencoes de figuras do repo (build_*_report_figures.py): import do
matplotlib dentro da funcao, dpi=150, salva .pdf e .png com bbox_inches="tight".

Dependencias:
    pip install matplotlib numpy

Uso:
    # Gera as tres figuras a partir de um run
    python tools/plot_gnss_drift.py dataset_gnss_denial/run_20260806_143000

    # Aponta o CSV diretamente e escolhe o prefixo de saida
    python tools/plot_gnss_drift.py --csv path/to/telemetry.csv --prefix gnss_run1
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"

# Paleta consistente com o resto do relatorio.
COLOR_TRUTH = "#2ECC71"   # ground truth Cosys-AirSim
COLOR_PX4 = "#E74C3C"     # estimativa EKF2
COLOR_DENIAL = "#34495E"  # marcador do instante de negacao


def load_run(csv_path: Path) -> dict[str, list[float]]:
    """Le o telemetry.csv em colunas de float (linhas invalidas sao descartadas)."""
    cols: dict[str, list[float]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                try:
                    val = float(v)
                except (TypeError, ValueError):
                    val = float("nan")
                cols.setdefault(k, []).append(val)
    if not cols:
        raise ValueError(f"CSV vazio ou ilegivel: {csv_path}")
    return cols


def _denial_time(cols: dict, meta: dict | None) -> float | None:
    """Instante (t_mono) da negacao de GNSS, do meta.json ou da coluna gps_denied."""
    if meta and meta.get("denial_t_mono") is not None:
        return float(meta["denial_t_mono"])
    flags, times = cols.get("gps_denied", []), cols.get("t_mono", [])
    for flag, t in zip(flags, times):
        if flag >= 0.5:
            return t
    return None


def _aligned_px4(cols: dict, meta: dict | None) -> tuple[list, list, list]:
    """
    Estimativa do PX4 trazida para o referencial do AirSim.

    O CSV guarda os valores crus de cada fonte; o offset entre as origens (EKF vs.
    spawn) fica no meta.json. Sem aplicar esse offset as duas trajetorias apareceriam
    separadas por uma constante que nao tem nada a ver com deriva.
    """
    off = (meta or {}).get("origin_offset_px4_to_airsim") or (0.0, 0.0, 0.0)
    ox, oy, oz = (float(v) for v in off)
    return (
        [v + ox for v in cols["px4_x"]],
        [v + oy for v in cols["px4_y"]],
        [v + oz for v in cols["px4_z"]],
    )


def estimate_latency(cols: dict, meta: dict | None,
                     max_s: float = 0.6, step_s: float = 0.005) -> tuple[float, float, float]:
    """
    Estima a latencia de transporte do PX4 minimizando o RMS do erro.

    Procura o atraso tau tal que a verdade em (t - tau) melhor explique a
    estimativa lida em t. So faz sentido rodar num voo de REFERENCIA (GNSS ligado):
    com GNSS negado a deriva real seria absorvida pelo ajuste, inflando tau e
    escondendo justamente o efeito que o experimento quer medir.

    Retorna (tau_s, rms_no_tau, rms_com_tau).
    """
    t = cols["t_mono"]
    off = (meta or {}).get("origin_offset_px4_to_airsim") or (0.0, 0.0, 0.0)
    ox, oy, oz = (float(v) for v in off)

    # Usa so o trecho com GNSS ativo, se o run tiver as duas fases.
    flags = cols.get("gps_denied", [0.0] * len(t))
    idx = [i for i, f in enumerate(flags) if f < 0.5] or list(range(len(t)))

    def _interp(xs: list[float], q: float) -> float:
        if q <= t[0]:
            return xs[0]
        if q >= t[-1]:
            return xs[-1]
        lo, hi = 0, len(t) - 1
        while hi - lo > 1:
            m = (lo + hi) // 2
            if t[m] <= q:
                lo = m
            else:
                hi = m
        f = (q - t[lo]) / (t[hi] - t[lo])
        return xs[lo] + f * (xs[hi] - xs[lo])

    def _rms(tau: float) -> float:
        s, n = 0.0, 0
        for i in idx:
            q = t[i] - tau
            if q < t[0]:
                continue
            ex = _interp(cols["cosys_x"], q) - (cols["px4_x"][i] + ox)
            ey = _interp(cols["cosys_y"], q) - (cols["px4_y"][i] + oy)
            ez = _interp(cols["cosys_z"], q) - (cols["px4_z"][i] + oz)
            s += ex * ex + ey * ey + ez * ez
            n += 1
        return math.sqrt(s / n) if n else float("nan")

    steps = int(max_s / step_s) + 1
    best_rms, best_tau = min((_rms(k * step_s), k * step_s) for k in range(steps))
    return best_tau, _rms(0.0), best_rms


def build_trajectory_3d(cols: dict, meta: dict | None, outfile: Path) -> None:
    """Trajetoria 3D: verdade do simulador vs. estimativa do EKF2."""
    import matplotlib.pyplot as plt

    px, py, pz = _aligned_px4(cols, meta)
    cx, cy, cz = cols["cosys_x"], cols["cosys_y"], cols["cosys_z"]

    fig = plt.figure(figsize=(9.5, 6), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    # Z NED aponta para baixo; plotamos -Z para que "para cima" seja altitude.
    ax.plot(cx, cy, [-v for v in cz], color=COLOR_TRUTH, linewidth=2,
            label="Ground truth (Cosys-AirSim)")
    ax.plot(px, py, [-v for v in pz], color=COLOR_PX4, linewidth=2,
            linestyle="--", label="Estimativa EKF2 (PX4 SITL)")

    t_deny = _denial_time(cols, meta)
    if t_deny is not None:
        i = min(range(len(cols["t_mono"])), key=lambda k: abs(cols["t_mono"][k] - t_deny))
        ax.scatter([cx[i]], [cy[i]], [-cz[i]], color=COLOR_DENIAL, s=70, marker="X",
                   zorder=5, label="Negacao de GNSS")

    ax.set_xlabel("Norte X (m)", fontsize=10)
    ax.set_ylabel("Leste Y (m)", fontsize=10)
    ax.set_zlabel("Altitude (m)", fontsize=10)
    ax.set_title("Trajetoria 3D — estimativa inercial vs. verdade do simulador",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")

    fig.tight_layout()
    fig.savefig(outfile.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(outfile.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {outfile.stem}")


def build_top_view(cols: dict, meta: dict | None, outfile: Path) -> None:
    """Vista superior (XY) com o ponto de negacao destacado."""
    import matplotlib.pyplot as plt

    px, py, _ = _aligned_px4(cols, meta)
    cx, cy = cols["cosys_x"], cols["cosys_y"]

    fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=150)
    ax.plot(cx, cy, color=COLOR_TRUTH, linewidth=2, label="Ground truth (Cosys-AirSim)")
    ax.plot(px, py, color=COLOR_PX4, linewidth=2, linestyle="--",
            label="Estimativa EKF2 (PX4 SITL)")

    t_deny = _denial_time(cols, meta)
    if t_deny is not None:
        i = min(range(len(cols["t_mono"])), key=lambda k: abs(cols["t_mono"][k] - t_deny))
        ax.scatter([cx[i]], [cy[i]], color=COLOR_DENIAL, s=90, marker="X", zorder=5,
                   label=f"Negacao de GNSS (t={t_deny:.1f} s)")
        ax.annotate("GNSS OFF", (cx[i], cy[i]), textcoords="offset points",
                    xytext=(8, 10), fontsize=9, fontweight="bold", color=COLOR_DENIAL)

    ax.set_xlabel("Norte X (m)", fontsize=11)
    ax.set_ylabel("Leste Y (m)", fontsize=11)
    ax.set_title("Vista superior — deriva lateral apos o corte de GNSS",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.35)
    ax.set_aspect("equal", adjustable="datalim")

    fig.tight_layout()
    fig.savefig(outfile.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(outfile.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {outfile.stem}")


def build_error_vs_time(cols: dict, meta: dict | None, outfile: Path) -> None:
    """
    Erro de posicao vs. tempo — a figura principal do artigo.

    Mostra o erro aproximadamente plano enquanto o GNSS esta ativo e o crescimento
    apos o corte, que e a evidencia quantitativa da degradacao inercial.
    """
    import matplotlib.pyplot as plt

    t = cols["t_mono"]
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

    ax.plot(t, cols["err_norm"], color=COLOR_PX4, linewidth=2, label="Erro 3D |e|")
    ax.plot(t, [abs(v) for v in cols["err_x"]], color="#3498DB", linewidth=1.2,
            alpha=0.8, label="|e_x| (Norte)")
    ax.plot(t, [abs(v) for v in cols["err_y"]], color="#9B59B6", linewidth=1.2,
            alpha=0.8, label="|e_y| (Leste)")
    ax.plot(t, [abs(v) for v in cols["err_z"]], color="#F39C12", linewidth=1.2,
            alpha=0.8, label="|e_z| (vertical)")

    t_deny = _denial_time(cols, meta)
    if t_deny is not None:
        ax.axvline(t_deny, color=COLOR_DENIAL, linestyle="--", linewidth=1.5, alpha=0.8)
        ax.annotate("GNSS negado", (t_deny, ax.get_ylim()[1]), textcoords="offset points",
                    xytext=(6, -14), fontsize=9, fontweight="bold", color=COLOR_DENIAL)
        # Sombreia a fase com GNSS para separar visualmente baseline de deriva.
        ax.axvspan(t[0], t_deny, color=COLOR_TRUTH, alpha=0.07)
        ax.annotate("GNSS ativo", ((t[0] + t_deny) / 2, ax.get_ylim()[1]),
                    textcoords="offset points", xytext=(0, -14), ha="center",
                    fontsize=9, color="#27AE60")

    ax.set_xlabel("Tempo de voo (s)", fontsize=11)
    ax.set_ylabel("Erro de posicao (m)", fontsize=11)
    ax.set_title("Degradacao da odometria inercial do EKF2 apos negacao de GNSS",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.35)

    fig.tight_layout()
    fig.savefig(outfile.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(outfile.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {outfile.stem}")


def build_all_figures(csv_path: Path, meta_path: Path | None = None,
                      prefix: str | None = None) -> None:
    """Gera as tres figuras de um run. Chamada tambem pelo orquestrador ao final do voo."""
    cols = load_run(csv_path)
    meta = None
    if meta_path and Path(meta_path).is_file():
        try:
            meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[AVISO] meta.json ilegivel ({e}); seguindo sem alinhamento de origem.")

    slug = prefix or f"gnss_{csv_path.parent.name}"
    FIGURES.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Gerando figuras de {csv_path} ({len(cols['t_mono'])} amostras):")

    build_trajectory_3d(cols, meta, FIGURES / f"{slug}_trajectory_3d")
    build_top_view(cols, meta, FIGURES / f"{slug}_top_view")
    build_error_vs_time(cols, meta, FIGURES / f"{slug}_error_vs_time")

    # Resumo numerico para citar no texto do artigo.
    errs = cols["err_norm"]
    flags = cols.get("gps_denied", [0.0] * len(errs))
    with_gps = [e for e, f in zip(errs, flags) if f < 0.5]
    without = [e for e, f in zip(errs, flags) if f >= 0.5]
    summary = {
        "samples": len(errs),
        "err_norm_final_m": errs[-1] if errs else None,
        "err_norm_max_m": max(errs) if errs else None,
        "err_norm_mean_gps_on_m": sum(with_gps) / len(with_gps) if with_gps else None,
        "err_norm_mean_gps_off_m": sum(without) / len(without) if without else None,
        "duration_s": cols["t_mono"][-1] if cols.get("t_mono") else None,
    }
    out_json = FIGURES / f"{slug}_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  -> {out_json.name}")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"     {k:28s} = {v:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Figuras de deriva do experimento GNSS-denied")
    ap.add_argument("run_dir", nargs="?", type=Path,
                    help="Pasta do run (contendo telemetry.csv e meta.json)")
    ap.add_argument("--csv", type=Path, default=None, help="Caminho direto do telemetry.csv")
    ap.add_argument("--meta", type=Path, default=None, help="Caminho direto do meta.json")
    ap.add_argument("--prefix", default=None, help="Prefixo dos arquivos em figures/")
    ap.add_argument("--estimate-latency", action="store_true",
                    help="So estima a latencia de transporte (use num voo de referencia)")
    args = ap.parse_args()

    if args.csv:
        csv_path = args.csv
        meta_path = args.meta or csv_path.parent / "meta.json"
    elif args.run_dir:
        csv_path = args.run_dir / "telemetry.csv"
        meta_path = args.meta or args.run_dir / "meta.json"
    else:
        ap.error("informe run_dir ou --csv")

    if not csv_path.is_file():
        ap.error(f"nao encontrado: {csv_path}")

    if args.estimate_latency:
        cols = load_run(csv_path)
        meta = None
        if Path(meta_path).is_file():
            meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        tau, rms0, rms_tau = estimate_latency(cols, meta)
        print(f"Latencia de transporte estimada: {tau * 1000:.0f} ms")
        print(f"  RMS sem correcao : {rms0:.4f} m")
        print(f"  RMS com correcao : {rms_tau:.4f} m")
        print(f"\nUse com: --latency-s {tau:.3f}")
        return

    build_all_figures(csv_path, meta_path, args.prefix)


if __name__ == "__main__":
    main()
