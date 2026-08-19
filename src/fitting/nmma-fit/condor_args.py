"""Print the fit_nmma.py CLI args for one object, for HTCondor submission.

Lives here (not condor/) because only nmma-fit knows its own flags
(--ebv-max, --z-known, --model, --prior) and how to compute --ebv-max
(extinction.get_sfd_ebv x EBV_SF2011). condor/submit_batch.py calls this
generically for any backend, without knowing backend-specific argument
names or how to build them.

Prints one argument per line to stdout, so the caller never has to worry
about shell-quoting values that might contain spaces.

Usage:
    python condor_args.py --object SN2021ugl --ra 246.9820265 --dec 20.253166509091 \\
        --model v19-1993j-corr --prior config/priors/sncosmo-generic.prior \\
        --data-dir data/processed --outdir results/ [--z-known 0.0412]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ingest"))
from extinction import EBV_SF2011, get_sfd_ebv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--object", required=True)
    parser.add_argument("--ra", type=float, required=True)
    parser.add_argument("--dec", type=float, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prior", required=True)
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--outdir", default="results/")
    parser.add_argument("--z-known", type=float, default=None, dest="z_known")
    return parser.parse_args()


def build_fit_args(args: argparse.Namespace) -> list[str]:
    ebv_max = get_sfd_ebv(args.ra, args.dec) * EBV_SF2011
    fit_args = [
        "--object",
        args.object,
        "--ra",
        str(args.ra),
        "--dec",
        str(args.dec),
        "--data",
        str(Path(args.data_dir) / f"{args.object}.dat"),
        "--model",
        args.model,
        "--prior",
        args.prior,
        "--ebv-max",
        str(ebv_max),
        "--outdir",
        args.outdir,
    ]
    if args.z_known is not None:
        fit_args += ["--z-known", str(args.z_known)]
    return fit_args


def main() -> None:
    print("\n".join(build_fit_args(parse_args())))


if __name__ == "__main__":
    main()
