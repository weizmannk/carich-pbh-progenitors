"""Fit M_Ni (and M_ej) for a single ZTF object with redback.

Primary pipeline. One job per object (HTCondor), no MPI. Sampler: dynesty via bilby.

Usage:
    python fit_nickel.py --object ZTF20abaunmw --kappa 0.07 --outdir results/
"""

import argparse
import logging

import matplotlib

matplotlib.use("Agg")

import redback

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--object", required=True, help="ZTF object name (e.g. ZTF20abaunmw)"
    )
    parser.add_argument(
        "--data", required=True, help="Path to the photometry in redback format"
    )
    parser.add_argument("--priors", required=True, help="YAML/bilby prior file")
    parser.add_argument(
        "--kappa",
        type=float,
        required=True,
        help="Fixed opacity (cm^2/g): 0.05, 0.07, or 0.10",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory (posteriors, corner plots)",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    logger.info("redback version: %s", redback.__version__)
    logger.info("seed: %s", args.seed)
    raise NotImplementedError("redback chain not yet validated on ZTF20abaunmw.")


if __name__ == "__main__":
    main()
