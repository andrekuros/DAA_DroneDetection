"""Unified CLI: gnss-bench <subcommand> ..."""

from __future__ import annotations

import argparse
import sys


def _delegate(module_main) -> None:
    """Drop the subcommand so the module argparse sees only its own flags."""
    sys.argv = [sys.argv[0], *sys.argv[2:]]
    module_main()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="gnss-bench",
        description="GNSS-denied lockstep bench (Cosys-AirSim + PX4 SITL)",
    )
    parser.add_argument(
        "command",
        choices=("fly", "campaign", "analyze", "vio", "fg", "plot", "probe-corridor"),
        help="Pipeline step",
    )
    parser.add_argument("rest", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    # Parse only the subcommand; remainder stays in sys.argv for the module.
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        parser.print_help()
        return
    cmd = sys.argv[1]
    if cmd == "fly":
        from gnss_bench.experiment import main as fly_main
        _delegate(fly_main)
    elif cmd == "campaign":
        from gnss_bench.campaign import main as campaign_main
        _delegate(campaign_main)
    elif cmd == "analyze":
        from gnss_bench.analyze import main as analyze_main
        _delegate(analyze_main)
    elif cmd == "vio":
        from gnss_bench.vio_frontend import main as vio_main
        _delegate(vio_main)
    elif cmd == "fg":
        from gnss_bench.factor_graph import main as fg_main
        _delegate(fg_main)
    elif cmd == "plot":
        from gnss_bench.plot_drift import main as plot_main
        _delegate(plot_main)
    elif cmd == "probe-corridor":
        from gnss_bench.corridor import main as probe_main
        _delegate(probe_main)
    else:
        parser.error(f"unknown command: {cmd}")


if __name__ == "__main__":
    main()
