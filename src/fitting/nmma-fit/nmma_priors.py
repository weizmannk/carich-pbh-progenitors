"""NMMA prior configuration: timeshift + D_L safeguard (nmma>=1.0.1).

Always applied, even without --timeshift: patching luminosity_distance.minimum
to be > 0 avoids a CosmologyError in check_model_time_consistency (z_at_value
is undefined for D_L <= 0 Mpc).

--z-known: known spectroscopic redshift (TNS). Do not trust the raw redshift
at low z without cross-checking against NED/Tully-Fisher (peculiar velocities
dominate). This module applies the given value as-is; that check happens
upstream (src/ingest/).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Union

import bilby

Timeshift = Optional[Union[float, Dict[str, float]]]


def create_prior(
    prior_file: Path, timeshift: Timeshift, outdir: Path, label: str = "prior"
) -> Path:
    """Load a prior file and apply the timeshift + D_L safeguard. Writes
    nothing if no modification is needed (returns the original path)."""
    if not os.path.isfile(prior_file):
        raise FileNotFoundError(f"Prior file not found: {prior_file}")

    priors = bilby.gw.prior.PriorDict(str(prior_file))
    modified = False

    if "luminosity_distance" in priors:
        dl = priors["luminosity_distance"]
        dl_min = getattr(dl, "minimum", None)
        if dl_min is not None and float(dl_min) <= 0:
            dl.minimum = 1e-3
            modified = True
            logging.debug("luminosity_distance.minimum <= 0, patched -> 1e-3 Mpc")

    if "timeshift" not in priors:
        logging.warning("No 'timeshift' parameter in prior file %s.", prior_file)
    elif timeshift is None:
        logging.debug(
            "No timeshift provided (explosion time unconstrained), original prior kept."
        )
    else:
        ts = timeshift
        if isinstance(ts, dict):
            if "min" in ts and "max" in ts:
                new_min, new_max = float(ts["min"]), float(ts["max"])
                if (
                    getattr(priors["timeshift"], "minimum", None) != new_min
                    or getattr(priors["timeshift"], "maximum", None) != new_max
                ):
                    priors["timeshift"].minimum = new_min
                    priors["timeshift"].maximum = new_max
                    modified = True
                    logging.debug("timeshift -> Uniform(%s, %s)", new_min, new_max)
            elif "mu" in ts and "sigma" in ts:
                new_mu, new_sigma = float(ts["mu"]), float(ts["sigma"])
                if (
                    getattr(priors["timeshift"], "mu", None) != new_mu
                    or getattr(priors["timeshift"], "sigma", None) != new_sigma
                ):
                    priors["timeshift"].mu = new_mu
                    priors["timeshift"].sigma = new_sigma
                    modified = True
                    logging.debug(
                        "timeshift -> Gaussian(mu=%s, sigma=%s)", new_mu, new_sigma
                    )
            elif "peak" in ts:
                new_peak = float(ts["peak"])
                if getattr(priors["timeshift"], "peak", None) != new_peak:
                    priors["timeshift"] = bilby.core.prior.DeltaFunction(
                        peak=new_peak, name="timeshift"
                    )
                    modified = True
                    logging.debug("timeshift -> DeltaFunction(peak=%s)", new_peak)
            else:
                logging.warning("Unrecognized timeshift keys: %s", list(ts.keys()))
        elif isinstance(ts, (int, float)):
            if (
                getattr(priors["timeshift"], "minimum", None) != ts
                or getattr(priors["timeshift"], "maximum", None) != ts
            ):
                priors["timeshift"] = bilby.core.prior.DeltaFunction(
                    peak=float(ts), name="timeshift"
                )
                modified = True
                logging.debug("timeshift -> fixed %s", ts)
        else:
            logging.warning("Unsupported timeshift type: %s", type(ts))

    if not modified:
        return Path(prior_file)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    priors.to_file(outdir=str(outdir), label=label)
    return outdir / f"{label}.prior"


def _dl_from_z(z: float) -> float:
    from astropy.cosmology import Planck18

    return float(Planck18.luminosity_distance(z).to("Mpc").value)


def configure_prior(
    prior_file: Path,
    outdir: Path,
    label: str,
    timeshift: Timeshift = None,
    z_known: Optional[float] = None,
) -> Path:
    """Apply the timeshift + D_L safeguard (always), then fix D_L via z_known if given.

    supernova_mag_boost is never touched here, with or without z_known.
    """
    prior_path = create_prior(
        prior_file=prior_file, timeshift=timeshift, outdir=outdir, label=label
    )

    if z_known is None:
        return Path(prior_path)

    priors = bilby.gw.prior.PriorDict(str(prior_path))
    priors["luminosity_distance"] = bilby.core.prior.DeltaFunction(
        peak=_dl_from_z(z_known),
        name="luminosity_distance",
        latex_label="$D_L$",
    )
    out_label = f"{label}_zfixed"
    priors.to_file(outdir=str(outdir), label=out_label)
    out_path = Path(outdir) / f"{out_label}.prior"
    logging.info("Prior configured (z_known=%s, D_L fixed) -> %s", z_known, out_path)
    return out_path
