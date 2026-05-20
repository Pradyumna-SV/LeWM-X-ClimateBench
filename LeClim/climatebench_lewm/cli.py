"""CLI: NetCDF ClimateBench → SWM HDF5 for official LeWM training."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from climatebench_lewm.convert import (
    default_output_path,
    write_climatebench_hdf5,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--climatebench-root",
        default=os.environ.get("CLIMATEBENCH_ROOT"),
        help="Directory with inputs_*.nc / outputs_*.nc (or env CLIMATEBENCH_ROOT)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .h5 path (default: $STABLEWM_HOME/<dataset-name>.h5)",
    )
    parser.add_argument(
        "--dataset-name",
        default="climatebench_train",
        help="Stem used under STABLEWM_HOME if --output omitted",
    )
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--co2-max", type=float, default=9500.0)
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help="Optional experiment ids (default: all paired files)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output file if it exists",
    )
    parser.add_argument(
        "--include-ch4",
        action="store_true",
        help="Append normalized CH4 (/ ch4-max) as second action column (retrain LeWM with action_dim=2)",
    )
    parser.add_argument(
        "--ch4-max",
        type=float,
        default=0.8,
        help="CH4 normalizer (ClimateBench-style default 0.8)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    if not args.climatebench_root:
        print("Set CLIMATEBENCH_ROOT or pass --climatebench-root", file=sys.stderr)
        sys.exit(2)

    out = Path(args.output) if args.output else default_output_path(args.dataset_name)

    write_climatebench_hdf5(
        args.climatebench_root,
        out,
        experiments=args.experiments,
        img_size=args.img_size,
        co2_max=args.co2_max,
        overwrite=args.overwrite,
        include_ch4=args.include_ch4,
        ch4_max=args.ch4_max,
    )
    print(out)
