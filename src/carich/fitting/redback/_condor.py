"""Print the ``_fit`` CLI args for one object, for HTCondor submission.

Lives here (not condor/) because only this backend knows its own flags.
condor/submit_batch.py calls this generically for any backend, without
knowing backend-specific argument names or how to build them.

Simpler than the NMMA counterpart: there is no --ebv-max to pre-compute,
because ``_fit`` queries the Planck GNILC map itself from --ra/--dec
(and falls back to no correction if the map is unreachable). The coordinates
are still forwarded so a worker with no targets.csv can do that query.

Prints one argument per line to stdout, so the caller never has to worry
about shell-quoting values that might contain spaces.

Usage:
    python -m carich.fitting.redback._condor --object SN2021ugl --ra 246.9820265 --dec 20.253166509091 \\
        --model arnett --data-dir data/processed --outdir results/ \\
        [--z-known 0.0412] [--kappa 0.07] [--priors priors_redback/arnett.prior]
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--object", required=True)
    parser.add_argument("--ra", type=float, default=None)
    parser.add_argument("--dec", type=float, default=None)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-dir", default="data/processed", dest="data_dir")
    parser.add_argument("--outdir", default="results/")
    parser.add_argument("--z-known", type=float, default=None, dest="z_known")
    parser.add_argument("--kappa", type=float, default=None)
    parser.add_argument("--priors", default=None)
    parser.add_argument("--nlive", type=int, default=None)
    parser.add_argument("--sampler", default=None)
    parser.add_argument("--error-budget", type=float, default=None, dest="error_budget")
    parser.add_argument("--data-tmin", type=float, default=None, dest="data_tmin")
    parser.add_argument("--data-tmax", type=float, default=None, dest="data_tmax")
    return parser.parse_args()


def build_fit_args(args: argparse.Namespace) -> list[str]:
    fit_args = [
        "--object",
        args.object,
        "--model",
        args.model,
        "--data",
        str(Path(args.data_dir) / f"{args.object}.dat"),
        "--outdir",
        args.outdir,
    ]
    # Everything below is optional in _fit.py: pass a flag only when a
    # value was given, so the script's own defaults stay in one place.
    for flag, value in (
        ("--ra", args.ra),
        ("--dec", args.dec),
        ("--z-known", args.z_known),
        ("--kappa", args.kappa),
        ("--priors", args.priors),
        ("--nlive", args.nlive),
        ("--sampler", args.sampler),
        ("--error-budget", args.error_budget),
        ("--data-tmin", args.data_tmin),
        ("--data-tmax", args.data_tmax),
    ):
        if value is not None:
            fit_args += [flag, str(value)]
    return fit_args


def main() -> None:
    print("\n".join(build_fit_args(parse_args())))


if __name__ == "__main__":
    main()
