"""
vio_frontend.py — Odometria visual monocular sobre os frames nadir
===================================================================
Consome `frames.csv` + `frames/*.jpg` de um run e produz `vio_odom.csv`: para
cada par de keyframes, a rotacao relativa e a DIRECAO unitaria da translacao.

Monocular nao observa escala. Isso e deliberado: a escala entra no factor graph
pela IMU e pelo barometro (ver run_factor_graph_replay.py). O que a camera
contribui e a direcao do movimento e a rotacao, que e justamente o que a IMU
integra mal.

Keyframe por DESLOCAMENTO, nao por tempo
-----------------------------------------
A 250 m AGL, com o veiculo a ~6 m/s e frames a 4 Hz, dois frames consecutivos
distam 1,5 m. A razao baseline/profundidade fica em 1/167, e nesse regime a
matriz essencial degenera: a translacao vira ruido e `recoverPose` devolve
direcao aleatoria. Por isso acumulamos frames ate o deslocamento passar de
`--min-baseline-m` (padrao 20 m, razao ~0,08) antes de fechar um par.

O deslocamento e medido pelo GT gravado no frames.csv. Isso e honesto: serve so
para ESCOLHER quais frames casar, nao entra na estimativa — a direcao sai da
imagem. Sem GT o mesmo criterio sairia da IMU integrada.

Dependencias:
    pip install opencv-python numpy

Uso:
    # Um run
    python tools/vio_frontend.py dataset_gnss_denial/campaign_vio250/denied_rep01

    # Todos os runs de uma campanha
    python tools/vio_frontend.py dataset_gnss_denial/campaign_vio250 --all

    # Diagnostico: salva imagens com os tracks desenhados
    python tools/vio_frontend.py <run> --debug-dir figures/vio_debug
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np

# Intrinsecos da vio_cam: FOV horizontal 90 deg em 1024x768.
# fx = (W/2) / tan(FOV/2) = 512 / tan(45) = 512
CAM_W, CAM_H = 1024, 768
FX = FY = 512.0
CX, CY = CAM_W / 2.0, CAM_H / 2.0
K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float64)

OUT_FIELDS = [
    "kf_from", "kf_to",          # indices dos frames casados
    "t_from", "t_to",            # t_wall (mesma base do telemetry.csv)
    "baseline_gt_m",             # deslocamento GT entre os keyframes (criterio de selecao)
    "tx", "ty", "tz",            # direcao UNITARIA da translacao, no frame da camera
    "rx", "ry", "rz",            # rotacao relativa como vetor de Rodrigues
    "n_matches", "n_inliers",    # qualidade -> peso do fator no factor graph
    "inlier_ratio",
    "baseline_m",                # translacao METRICA (escala do LiDAR); nan se indisponivel
    "agl_m",                     # alcance nadir usado para escalar
    "dir_err_deg",               # auto-verificacao vs verdade (nao usado na estimativa)
    "scale_err_m",               # auto-verificacao da escala vs verdade
]


def load_index(run_dir: Path) -> list[dict]:
    """Le frames.csv, ordenado por tempo."""
    idx_path = run_dir / "frames.csv"
    if not idx_path.exists():
        raise FileNotFoundError(
            f"{idx_path} nao existe. O run foi gravado com --record-frames?"
        )
    with idx_path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{idx_path} esta vazio.")
    rows.sort(key=lambda r: float(r["t_wall"]))
    return rows


def _gt_xyz(row: dict) -> np.ndarray:
    """Posicao GT do frame. Aceita os nomes usados pelo FrameRecorder."""
    for tri in (("gt_x", "gt_y", "gt_z"), ("cosys_x", "cosys_y", "cosys_z"),
                ("x", "y", "z")):
        if all(k in row and row[k] not in ("", None) for k in tri):
            return np.array([float(row[tri[0]]), float(row[tri[1]]),
                             float(row[tri[2]])], dtype=np.float64)
    raise KeyError(f"frames.csv sem colunas de posicao GT; colunas: {list(row)}")


def _frame_path(run_dir: Path, row: dict) -> Path:
    """
    Caminho do JPEG do frame.

    `frame_file` e o nome que o FrameRecorder grava; os outros ficam por
    tolerancia. Sem ele o fallback por indice devolvia f_000000.jpg para TODOS
    os frames, e casar uma imagem com ela mesma da translacao nula — o
    front-end rejeitava 100% dos pares sem dizer por que.
    """
    for key in ("frame_file", "frame", "file", "path", "filename"):
        if key in row and row[key]:
            p = Path(row[key])
            return p if p.is_absolute() else (run_dir / p)
    raise KeyError(
        f"frames.csv sem coluna de caminho de imagem; colunas: {list(row)}"
    )


def _Rx(a): c, s = math.cos(a), math.sin(a); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
def _Ry(a): c, s = math.cos(a), math.sin(a); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
def _Rz(a): c, s = math.cos(a), math.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def body_to_ned(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Rotacao corpo->NED na convencao ZYX."""
    return _Rz(yaw) @ _Ry(pitch) @ _Rx(roll)


# Camera nadir montada no corpo (Pitch -90 no settings.json). Olhando para baixo
# com guinada zero, "para cima" na imagem aponta para o Norte:
#   x_cam = Leste, y_cam = -Norte, z_cam = Down
R_CAM_FROM_BODY = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64)


def cam_from_ned(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Rotacao NED->camera, dada a atitude do veiculo."""
    return R_CAM_FROM_BODY @ body_to_ned(roll, pitch, yaw).T


def estimate_pair(img_a: np.ndarray, img_b: np.ndarray,
                  R_rel: np.ndarray | None = None,
                  n_features: int = 3000,
                  ransac_px: float = 3.0) -> dict | None:
    """
    Rotacao + direcao da translacao entre dois frames, via matriz essencial.

    ORB com casamento por descritor, nao LK. Medido nestes dados: com fluxo
    mediano de ~110 px entre keyframes, o LK reporta sucesso mas erra o
    casamento, e o RANSAC aceitava 32 de 869 pontos — a estimativa era
    descartada em 100% dos pares. Com ORB o mesmo par da ~920 inliers.

    O limiar de RANSAC tambem importa: a 1 px quase nada passa; 3 px reflete o
    residuo real de reprojecao nesta cena e triplica os inliers.

    `R_rel` (rotacao relativa camera->camera, da atitude do veiculo) permite
    de-rotacionar o segundo frame antes de estimar E. Nao muda o erro mediano,
    mas corta o p90 de 37 para 18 graus: quando a rotacao domina o fluxo, E
    tenta explica-la como translacao e a direcao sai torta.

    Retorna None quando o par nao sustenta uma estimativa — melhor descartar o
    fator do que alimentar o factor graph com uma direcao inventada.
    """
    orb = cv2.ORB_create(nfeatures=n_features)
    ka, da = orb.detectAndCompute(img_a, None)
    kb, db = orb.detectAndCompute(img_b, None)
    if da is None or db is None or len(da) < 50 or len(db) < 50:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(da, db), key=lambda m: m.distance)[:1500]
    if len(matches) < 40:
        return None

    a = np.float32([ka[m.queryIdx].pt for m in matches])
    b_raw = np.float32([kb[m.trainIdx].pt for m in matches])

    if R_rel is not None:
        # Leva os pontos de b para o referencial angular de a.
        h = np.hstack([b_raw, np.ones((len(b_raw), 1), dtype=np.float32)])
        rays = np.linalg.inv(K) @ h.T
        rays = R_rel.T @ rays
        rays = rays / rays[2:3, :]
        b = (K @ rays)[:2].T.astype(np.float32)
    else:
        b = b_raw

    E, mask_e = cv2.findEssentialMat(
        a, b, K, method=cv2.RANSAC, prob=0.999, threshold=ransac_px,
    )
    if E is None or E.shape != (3, 3) or mask_e is None or int(mask_e.sum()) < 25:
        return None

    pose_mask = mask_e.copy()
    n_in, R, t, _ = cv2.recoverPose(E, a, b, K, mask=pose_mask)
    if n_in < 25:
        return None

    # Profundidade mediana dos inliers, em unidades de baseline.
    # Triangulando com baseline unitaria, a profundidade sai na mesma unidade
    # arbitraria da translacao. Comparada depois com o alcance medido pelo LiDAR
    # nadir, ela DA A ESCALA METRICA — que a camera sozinha nao observa e que e
    # justamente o que falta para corrigir deriva inercial.
    # A mediana tem de vir da MESMA regiao que o LiDAR mede. A camera abre 90 graus
    # (pegada de +-250 m a 250 m de altura) e o cone do LiDAR e de +-10 graus
    # (+-44 m): tomar a mediana sobre a imagem inteira compara duas populacoes
    # diferentes de profundidade e a escala sai enviesada. Restringimos ao cone
    # central equivalente.
    lidar_half_fov_deg = 10.0
    r_px = FX * math.tan(math.radians(lidar_half_fov_deg))

    depth_unit = float("nan")
    try:
        sel = pose_mask.reshape(-1).astype(bool)
        # so pontos dentro do cone central
        d_px = np.linalg.norm(a - np.array([CX, CY], dtype=np.float32), axis=1)
        sel = sel & (d_px <= r_px)
        if sel.sum() >= 10:
            P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
            P2 = K @ np.hstack([R, t.reshape(3, 1)])
            # ascontiguousarray e obrigatorio: `.T` devolve uma view nao
            # contigua e o binding do OpenCV a rejeita com um erro de
            # "missing required argument", que nao sugere nada sobre layout.
            pa = np.ascontiguousarray(a[sel].T, dtype=np.float64)
            pb = np.ascontiguousarray(b[sel].T, dtype=np.float64)
            X = cv2.triangulatePoints(P1, P2, pa, pb)
            w = X[3]
            good = np.abs(w) > 1e-9
            if good.sum() >= 10:
                Z = (X[2][good] / w[good])
                Z = Z[np.isfinite(Z) & (Z > 0)]
                if Z.size >= 10:
                    depth_unit = float(np.median(Z))
    except Exception:
        pass

    t = t.reshape(3)
    norm = np.linalg.norm(t)
    if not np.isfinite(norm) or norm < 1e-9:
        return None
    t = t / norm  # explicitamente unitario: a escala nao e observavel aqui

    rvec, _ = cv2.Rodrigues(R)
    return {
        "t_unit": t,
        "rvec": rvec.reshape(3),
        "n_matches": int(len(a)),
        "n_inliers": int(n_in),
        "inlier_ratio": float(n_in) / float(len(a)),
        "depth_unit": depth_unit,
        "pts_a": a,
        "pts_b": b,
    }


def _agl_from_cloud(run_dir: Path, row: dict, self_hit_m: float = 2.0) -> float:
    """
    Alcance nadir mediano da nuvem gravada = altura sobre a superficie dominante.

    Mediana, nao media: sobre a cidade a pegada do cone mistura rua e telhados, e
    a media escorrega para o que estiver mais espalhado. Descarta retornos abaixo
    de `self_hit_m`, que sao ecos na propria fuselagem.
    """
    name = row.get("cloud_file")
    if not name:
        return float("nan")
    p = Path(name)
    p = p if p.is_absolute() else (run_dir / p)
    if not p.is_file():
        return float("nan")
    try:
        pts = np.load(p)
        if pts.ndim != 2 or pts.shape[0] == 0:
            return float("nan")
        r = np.linalg.norm(pts[:, :3], axis=1)
        r = r[np.isfinite(r) & (r > self_hit_m)]
        return float(np.median(r)) if r.size >= 5 else float("nan")
    except Exception:
        return float("nan")


def _att(row: dict) -> tuple[float, float, float] | None:
    try:
        return (float(row["gt_roll"]), float(row["gt_pitch"]), float(row["gt_yaw"]))
    except (KeyError, ValueError, TypeError):
        return None


def process_run(run_dir: Path, min_baseline_m: float, debug_dir: Path | None,
                max_baseline_m: float) -> dict:
    rows = load_index(run_dir)
    out_path = run_dir / "vio_odom.csv"

    results: list[dict] = []
    n_rejected = 0

    anchor_i = 0
    anchor_row = rows[0]
    anchor_xyz = _gt_xyz(anchor_row)
    anchor_img = cv2.imread(str(_frame_path(run_dir, anchor_row)), cv2.IMREAD_GRAYSCALE)

    for j in range(1, len(rows)):
        row = rows[j]
        xyz = _gt_xyz(row)
        baseline = float(np.linalg.norm(xyz - anchor_xyz))
        if baseline < min_baseline_m:
            continue

        img = cv2.imread(str(_frame_path(run_dir, row)), cv2.IMREAD_GRAYSCALE)
        if anchor_img is None or img is None:
            anchor_i, anchor_row, anchor_xyz, anchor_img = j, row, xyz, img
            continue

        # De-rotacao pela atitude do veiculo. Nao e "usar a verdade": atitude
        # permanece observavel sem GNSS (giro + acel + magnetometro), e todo VIO
        # real acopla a atitude inercial. O que a camera fornece aqui e a
        # DIRECAO da translacao, que a IMU nao observa sem deriva.
        att_a, att_b = _att(anchor_row), _att(row)
        R_rel = None
        if att_a is not None and att_b is not None:
            R_rel = cam_from_ned(*att_b) @ cam_from_ned(*att_a).T

        est = estimate_pair(anchor_img, img, R_rel=R_rel)
        if est is None:
            n_rejected += 1
            # Baseline grande demais tambem quebra o tracking (LK perde os pontos).
            # Se ja passamos do teto, reancorar para nao travar o run inteiro.
            if baseline >= max_baseline_m:
                anchor_i, anchor_row, anchor_xyz, anchor_img = j, row, xyz, img
            continue

        # Auto-verificacao: angulo entre a direcao estimada e a verdadeira.
        # Nao entra na estimativa — e a metrica que diz se o front-end presta,
        # e o numero que o paper deve citar em vez de afirmar que "funciona".
        ang_err = float("nan")
        if att_a is not None:
            gt_dir = cam_from_ned(*att_a) @ (xyz - anchor_xyz)
            gn = np.linalg.norm(gt_dir)
            if gn > 1e-9:
                gt_dir = gt_dir / gn
                # |cos|: recoverPose tem ambiguidade de sinal na translacao
                c = abs(float(np.dot(est["t_unit"], gt_dir)))
                ang_err = math.degrees(math.acos(min(1.0, max(-1.0, c))))

        # ESCALA METRICA a partir do LiDAR nadir.
        #
        # A triangulacao com baseline unitaria da profundidades em unidades
        # arbitrarias. O LiDAR mede a mesma superficie em metros; a razao entre
        # as duas e o fator de escala. Sem isto o VIO so restringe direcao, e a
        # deriva inercial — que esta na MAGNITUDE — permanece intacta: medimos
        # ganho de apenas 1,4x no factor graph, contra 28x quando o fator trazia
        # deslocamento com escala.
        agl = _agl_from_cloud(run_dir, anchor_row)
        baseline_m = float("nan")
        du = est.get("depth_unit", float("nan"))
        if math.isfinite(agl) and math.isfinite(du) and du > 1e-6:
            baseline_m = agl / du  # |t| era unitario, entao a escala e o proprio fator
        scale_err = (baseline_m - baseline) if math.isfinite(baseline_m) else float("nan")

        results.append({
            "kf_from": anchor_i, "kf_to": j,
            "t_from": float(anchor_row["t_wall"]), "t_to": float(row["t_wall"]),
            "baseline_gt_m": baseline,
            "tx": est["t_unit"][0], "ty": est["t_unit"][1], "tz": est["t_unit"][2],
            "rx": est["rvec"][0], "ry": est["rvec"][1], "rz": est["rvec"][2],
            "n_matches": est["n_matches"], "n_inliers": est["n_inliers"],
            "inlier_ratio": est["inlier_ratio"],
            "baseline_m": baseline_m,
            "agl_m": agl,
            "dir_err_deg": ang_err,
            "scale_err_m": scale_err,
        })

        if debug_dir is not None:
            debug_dir.mkdir(parents=True, exist_ok=True)
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            for (pa, pb) in zip(est["pts_a"][:200], est["pts_b"][:200]):
                cv2.line(vis, tuple(pa.astype(int)), tuple(pb.astype(int)), (0, 200, 0), 1)
                cv2.circle(vis, tuple(pb.astype(int)), 2, (0, 0, 255), -1)
            cv2.imwrite(str(debug_dir / f"{run_dir.name}_kf{anchor_i:04d}_{j:04d}.jpg"), vis)

        anchor_i, anchor_row, anchor_xyz, anchor_img = j, row, xyz, img

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=OUT_FIELDS)
        w.writeheader()
        for r in results:
            w.writerow(r)

    ratios = [r["inlier_ratio"] for r in results]
    errs = [r["dir_err_deg"] for r in results if math.isfinite(r["dir_err_deg"])]
    scale_errs = [r["scale_err_m"] for r in results if math.isfinite(r["scale_err_m"])]
    stats = {
        "run": run_dir.name,
        "n_frames": len(rows),
        "n_pairs": len(results),
        "n_rejected": n_rejected,
        "median_inlier_ratio": float(np.median(ratios)) if ratios else float("nan"),
        "median_baseline_m": float(np.median([r["baseline_gt_m"] for r in results]))
        if results else float("nan"),
        "median_dir_err_deg": float(np.median(errs)) if errs else float("nan"),
        "p90_dir_err_deg": float(np.percentile(errs, 90)) if errs else float("nan"),
        "median_scale_err_m": (float(np.median(np.abs(scale_errs))) if scale_errs
                               else float("nan")),
        "n_scaled": len(scale_errs),
        "out": str(out_path),
    }
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="VIO monocular sobre os frames nadir")
    ap.add_argument("run_dir", type=Path,
                    help="Pasta do run (ou da campanha, com --all)")
    ap.add_argument("--all", action="store_true",
                    help="Processa todos os subdiretorios que tenham frames.csv")
    ap.add_argument("--min-baseline-m", type=float, default=20.0, metavar="M",
                    help="Deslocamento minimo entre keyframes (evita E degenerada)")
    ap.add_argument("--max-baseline-m", type=float, default=60.0, metavar="M",
                    help="Acima disso, reancora mesmo sem par valido")
    ap.add_argument("--debug-dir", type=Path, default=None,
                    help="Salva imagens com os tracks desenhados")
    args = ap.parse_args()

    if args.all:
        runs = sorted(p.parent for p in args.run_dir.glob("*/frames.csv"))
        if not runs:
            ap.error(f"Nenhum run com frames.csv em {args.run_dir}")
    else:
        runs = [args.run_dir]

    all_stats = []
    print(f"{'run':<20} {'frames':>7} {'pares':>6} {'rej':>5} {'inlier':>7} "
          f"{'base(m)':>8} {'erro(deg)':>10} {'p90':>6} {'esc(m)':>7} {'n_esc':>6}")
    print("-" * 92)
    for run in runs:
        try:
            s = process_run(run, args.min_baseline_m, args.debug_dir, args.max_baseline_m)
        except Exception as e:
            print(f"{run.name:<20} ERRO: {e}")
            continue
        print(f"{s['run']:<20} {s['n_frames']:>7} {s['n_pairs']:>6} {s['n_rejected']:>5} "
              f"{s['median_inlier_ratio']:>7.2f} {s['median_baseline_m']:>8.1f} "
              f"{s['median_dir_err_deg']:>10.1f} {s['p90_dir_err_deg']:>6.1f} "
              f"{s['median_scale_err_m']:>7.1f} {s['n_scaled']:>6}")
        all_stats.append(s)

    print("\n[OK] vio_odom.csv gravado em cada run.")
    print("     Direcao unitaria da translacao + rotacao; escala vem da IMU no factor graph.")


if __name__ == "__main__":
    main()
