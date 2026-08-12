# GNSS bench

Lockstep testbed for **GNSS-denied UAS navigation**: Cosys-AirSim (urban scene + sensors)
coupled to an unmodified **PX4 SITL** autopilot. GNSS aiding is revoked in flight;
every remaining stream is logged against simulator truth so you can develop and
score offline estimators (VIO, factor graphs, your own).

This package lives inside the DAA_DroneDetection monorepo. Detection/YOLO tools
are unchanged.

## Install

From the repo root, with the project venv active:

```powershell
pip install -e ./gnss_bench
```

Dependencies: `mavsdk`, `cosysairsim`, `numpy`, `scipy`, `opencv-python`, `matplotlib`.

## Pipeline

```
bring-up (Unreal + PX4)
        │
        ▼
 gnss-bench campaign   →  run_dir/telemetry.csv  (+ frames/, clouds/)
        │
        ▼
 gnss-bench vio        →  vio_odom.csv          (optional visual front-end)
        │
        ▼
 gnss-bench fg         →  factor-graph vs EKF2
        │
        ▼
 gnss-bench analyze    →  drift law + figures
```

| Command | What it does |
|---|---|
| `gnss-bench fly ...` | One flight (deny GNSS, log CSV) |
| `gnss-bench campaign ...` | N repetitions + PX4/Unreal lifecycle |
| `gnss-bench vio ...` | Monocular nadir VIO → `vio_odom.csv` |
| `gnss-bench fg ...` | Offline factor-graph replay |
| `gnss-bench analyze ...` | Campaign aggregate / drift law |
| `gnss-bench plot ...` | Per-run 3D / top / error figures |
| `gnss-bench probe-corridor ...` | Line-of-sight corridor check |

Equivalent: `python -m gnss_bench <same args>`. Old `python tools/run_gnss_*.py`
commands still work (thin shims).

## Quickstart (after sim is up)

```powershell
gnss-bench fly --dry-run
gnss-bench campaign --out-dir dataset_gnss_denial\my_campaign `
  --alt-m 250 --distance-m 250 --deny-at-m 15 `
  --baseline-reps 2 --reps 5 --corridor-e 0 `
  --record-frames --restart-every 5 --skip-existing
gnss-bench vio dataset_gnss_denial\my_campaign --all
gnss-bench fg dataset_gnss_denial\my_campaign --use-vio --prefix gnss_my_fg
gnss-bench analyze dataset_gnss_denial\my_campaign
```

Full bring-up, output contract, and how to plug in **your** estimator:
[docs/gnss_bench_guide.md](../docs/gnss_bench_guide.md).

Environment overrides (optional):

| Variable | Default |
|---|---|
| `GNSS_BENCH_SIM_EXE` | `D:\Projects\AirSim_Matrix\Windows\CitySample.exe` |
| `GNSS_BENCH_PX4_DIR` | `~/PX4-Autopilot` (WSL path) |
