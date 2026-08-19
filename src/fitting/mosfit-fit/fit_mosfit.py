"""Fit M_Ni (and M_ej) for a single ZTF object with MOSFiT.

Cross-validation only, on 2-3 objects. Runs in .venv-mosfit, a separate
interpreter from redback (MOSFiT 1.3 pins numpy<=1.26.4, redback requires
numpy 2.x). Writes to the same results/ directory with the same summary
JSON format as the redback fit, so the two remain directly comparable.

Usage:
    .venv-mosfit/bin/python fit_mosfit.py --object ZTF20abaunmw --outdir results/
"""

import argparse
import logging

import matplotlib

matplotlib.use("Agg")

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--object", required=True, help="ZTF object name (e.g. ZTF20abaunmw)"
    )
    parser.add_argument("--data", required=True, help="Path to the photometry")
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
    logger.info("seed: %s", args.seed)
    raise NotImplementedError(
        "MOSFiT cross-validation: not before the redback chain is validated."
    )


if __name__ == "__main__":
    main()
