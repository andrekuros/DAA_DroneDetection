"""Shim — import from `gnss_bench.px4` (see gnss_bench/README.md)."""
from gnss_bench.px4 import *  # noqa: F403
from gnss_bench.px4 import main

if __name__ == "__main__":
    main()
