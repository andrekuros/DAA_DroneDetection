"""Repo-root and environment paths for the GNSS bench."""

from __future__ import annotations

import os
from pathlib import Path

# gnss_bench/gnss_bench/paths.py -> parents[2] is the monorepo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
FIGURES = REPO_ROOT / "figures"
DEFAULT_SIM_EXE = r"D:\Projects\AirSim_Matrix\Windows\CitySample.exe"
DEFAULT_PX4_DIR = "~/PX4-Autopilot"


def sim_exe() -> str:
    return os.environ.get("GNSS_BENCH_SIM_EXE", DEFAULT_SIM_EXE)


def px4_dir() -> str:
    return os.environ.get("GNSS_BENCH_PX4_DIR", DEFAULT_PX4_DIR)


def settings_candidates() -> list[Path]:
    """AirSim settings.json locations, including the package copy."""
    home = Path.home()
    return [
        home / "Documents" / "AirSim" / "settings.json",
        home / "OneDrive" / "Documents" / "AirSim" / "settings.json",
        home / "OneDrive - Personal" / "Documents" / "AirSim" / "settings.json",
        PACKAGE_ROOT / "config" / "cosys_airsim_px4_settings.json",
        REPO_ROOT / "config" / "cosys_airsim_px4_settings.json",
    ]
