# GNSS bench — user guide

How to bring up the lockstep bench, log a campaign, and test an offline
navigation solution against EKF2 and simulator truth.

The installable package is [`gnss_bench/`](../gnss_bench/). Lab notes and
historical troubleshooting remain in [gnss_denial_setup.md](gnss_denial_setup.md).

## 1. Prerequisites

| Piece | Notes |
|---|---|
| Windows + WSL2 | PX4 does not build natively on Windows. Mirrored networking (`networkingMode=mirrored` in `.wslconfig`) so `127.0.0.1` is shared. |
| PX4 SITL | Clone `PX4-Autopilot` in WSL; `make px4_sitl_default none_iris`. Default tree: `~/PX4-Autopilot` (`GNSS_BENCH_PX4_DIR`). |
| Cosys-AirSim / CitySample | Unreal binary; default `GNSS_BENCH_SIM_EXE`. |
| Python venv | Repo `venv`; `pip install -e ./gnss_bench`. |

Copy PX4 vehicle settings (not the DAA/SimpleFlight `settings.json`):

```powershell
Copy-Item gnss_bench\config\cosys_airsim_px4_settings.json `
  "$env:USERPROFILE\OneDrive\Documents\AirSim\settings.json"
```

If your AirSim folder is `%USERPROFILE%\Documents\AirSim`, use that instead.
Restart the Unreal environment after changing `settings.json`.

**Ports:** MAVSDK talks to PX4 on **UDP 14550** (GCS). AirSim owns 14540
(`ControlPortLocal`). Lockstep TCP is **4560**.

## 2. Boot order

1. Start CitySample/Unreal and wait until the map is loaded (listens on TCP 4560).
   ```powershell
   & "$env:GNSS_BENCH_SIM_EXE" -windowed -ResX=1280 -ResY=720
   ```
   If the env var is unset, use your CitySample path.
2. PX4: either start it yourself
   ```bash
   wsl -e bash -lc "cd ~/PX4-Autopilot && PX4_SIM_HOSTNAME=127.0.0.1 make px4_sitl_default none_iris"
   ```
   and wait for `Simulator connected`, **or** let `gnss-bench campaign` start it.
3. Run commands from Windows with the venv active.

Do **not** restart PX4 alone while AirSim lockstep is up — the new SITL prints
“Simulator connected” but gets no IMU. Restart Unreal + PX4 together
(`--restart-every N` on the campaign does that).

## 3. Smoke checklist

```powershell
gnss-bench fly --dry-run
gnss-bench fly --no-fly --watch-s 10
gnss-bench fly --deny-at-m -1 --alt-m 250 --distance-m 80 --corridor-e 0
gnss-bench fly --deny-at-m 15 --alt-m 250 --distance-m 80 --corridor-e 0 --record-frames
```

Expect: EKF health OK; baseline `err_norm` small and stable; denied run sets
`gps_denied=1` after `--deny-at-m`. Optional: `gnss-bench probe-corridor --altitudes 250`.

## 4. Campaign

```powershell
gnss-bench campaign `
  --out-dir dataset_gnss_denial\my_campaign `
  --alt-m 250 --distance-m 250 --deny-at-m 15 `
  --corridor-e 0 `
  --baseline-reps 2 --reps 10 `
  --record-frames --frame-hz 4 --cloud-hz 2 `
  --restart-every 5 --max-retries 15 --skip-existing
```

Useful flags: `--skip-existing` resumes without overwriting good runs;
`--restart-every 5` renews Unreal+PX4 (MAVSDK param client times out on long
sessions); `--record-frames` is required for VIO.

For paper-style statistics, keep denied windows comparable (about 20–40 s).
Longer windows (failed land / timeout) inflate terminal error and should be
dropped before quoting medians.

## 5. How to test your solution

Each run folder is the unit of work:

```
run_dir/
  telemetry.csv          # required — EKF, GT, IMU, baro, mag, LiDAR stats
  meta.json              # config, denial, collision, recording counts
  frames.csv             # if --record-frames
  frames/*.jpg
  clouds/*.npy
  vio_odom.csv           # after gnss-bench vio
  my_estimate.csv        # your output (see below)
```

### Path A — reuse the bundled factor graph

1. Log with `--record-frames`.
2. `gnss-bench vio <campaign> --all` writes `vio_odom.csv` (unit translation
   direction + rotation; scale is **not** observable from the monocular pair).
3. `gnss-bench fg <campaign> --use-vio --prefix my_fg` fuses IMU + baro + VIO
   (+ LiDAR AGL) and writes figures under `figures/`.

Replace or wrap `gnss_bench/gnss_bench/factor_graph.py` / `vio_frontend.py` if
you want a different smoother or front-end; keep the same CSV inputs.

### Path B — your own estimator

Read `telemetry.csv` (and frames if you need vision). Write NED positions:

```
t,x,y,z
```

`t` should match `t_mono` or `timestamp` in the log; `x,y,z` are North, East,
Down (metres), same frame as `cosys_*` after the origin offset in `meta.json`.

Then:

```powershell
gnss-bench plot <run_dir>
```

overlays EKF vs truth. To overlay `my_estimate.csv`, add a column-compatible
CSV or extend the plot script — v1 does not register plugins. Compare
`err_norm` against EKF on `t_since_denial >= 0` only.

### `telemetry.csv` columns (contract)

| Group | Columns |
|---|---|
| Time | `timestamp`, `t_mono`, `t_since_denial`, `px4_age_s` |
| EKF2 | `px4_x`, `px4_y`, `px4_z` |
| Truth | `cosys_x`, `cosys_y`, `cosys_z` |
| Error | `err_x`, `err_y`, `err_z`, `err_norm`, `err_raw_norm` |
| Flag | `gps_denied` (0/1) |
| IMU | `imu_ax`…`imu_az`, `imu_gx`…`imu_gz` |
| Baro / mag | `baro_alt_m`, `baro_pressure`, `mag_x`…`mag_z` |
| LiDAR | `lidar_points`, `lidar_mean_range_m`, `lidar_min_range_m`, `lidar_agl_m` |
| Validity | `collided`, `collision_object` |

Coordinates are NED, metres. `err_*` are latency-compensated (default
`--latency-s 0.255` on this bench; re-estimate with
`gnss-bench plot <run> --estimate-latency` on a GNSS-on flight).

## 6. Pitfalls

- **C: disk / pagefile.** Unreal dies with “paging file is too small” when C:
  has ~1 GB free. Logs and frames go to W:; keep several GB free on C:.
- **`EKF2_GPS_CTRL` persistence.** Denial writes 0 to disk. `shutdown()` and
  `ensure_gnss_enabled()` restore it; if a session is poisoned:
  `rm ~/PX4-Autopilot/build/px4_sitl_default/rootfs/parameters*.bson` then restart PX4.
- **MAVSDK param TIMEOUT** on long sessions — use `--restart-every 5` (full
  Unreal+PX4). Restarting PX4 alone breaks lockstep.
- **Orphan `mavsdk_server.exe`** on 14550 looks like “EKF did not converge”.
- **Collisions** abort the run; campaign retries. Probe the corridor first.
- **Do not mix DAA and GNSS `settings.json`.** GNSS needs `PX4Multirotor` +
  `LockStep: true` + `ClockSpeed: 1.0`.
