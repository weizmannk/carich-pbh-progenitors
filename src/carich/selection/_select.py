from pathlib import Path

import numpy as np
import pandas as pd
from astropy import units as u

from ._classes import HYDROGEN_POOR, group
from ._hosts import associate
from ._lightcurve import interpolate, peak, read_alerts
from ._luminosity import (
    band_extinction,
    bandpass,
    distance_modulus,
    distance_modulus_error,
)
from ._nickel import bolometric_correction, luminosity, nickel_mass

RISE_TIMES = (10, 15) * u.day
"""Rise times bracketing the first-look nickel mass."""


def _extinction(band: str, Ebv: float) -> float:
    if not Ebv > 0:
        return 0.0
    return band_extinction(*bandpass(f"ztf{band}"), Ebv=Ebv)


def _measure(lightcurve: pd.DataFrame, redshift: float, Ebv: float) -> dict:
    """Peak, timescales, color and first-look yields of one light curve."""
    bands = {band: table for band, table in lightcurve.groupby("filter")}
    peaks = {
        band: peak(table["jd"], table["magpsf"], table["sigmapsf"])
        for band, table in bands.items()
    }
    reference = "r" if "r" in bands and len(bands["r"]) >= 2 else "g"
    if peaks.get(reference) is None:
        return {}
    best = peaks[reference]
    modulus = distance_modulus(redshift)
    stretch = 1 + redshift

    result = {
        "band": reference,
        "n_detections": int(lightcurve["filter"].isin(["g", "r"]).sum()),
        "peak_time": best.time,
        "peak_bracketed": best.bracketed,
        "peak_absolute_magnitude": best.magnitude
        - modulus
        - _extinction(reference, Ebv),
        "peak_absolute_magnitude_error": float(
            np.hypot(best.error, distance_modulus_error(redshift))
        ),
        "rise_time_min": (best.time - lightcurve["jd"].min()) / stretch,
    }

    table = bands[reference]
    later = interpolate(
        table["jd"], table["magpsf"], best.time + 10 * stretch, max_gap=12
    )
    result["decline_10d"] = later - best.magnitude

    if "g" in bands and peaks.get("r") is not None:
        r = peaks["r"]
        g = interpolate(bands["g"]["jd"], bands["g"]["magpsf"], r.time, max_gap=6)
        color = (g - _extinction("g", Ebv)) - (r.magnitude - _extinction("r", Ebv))
        result["color"] = color
        bolometric = g - modulus - _extinction("g", Ebv) + bolometric_correction(color)
        if np.isfinite(bolometric):
            masses = nickel_mass(luminosity(bolometric), RISE_TIMES).to_value(u.Msun)
            result["nickel_mass_min"], result["nickel_mass_max"] = masses
    return result


def _label(row: dict, max_dz, max_distance, faint_limit, min_detections) -> tuple:
    """Status of one object and, if it is not a candidate, the reason."""
    if not row["redshift"] > 0:
        return "rejected", "no redshift"
    if not np.isfinite(row.get("dz", np.nan)):
        return "rejected", "no matched galaxy"
    if abs(row["dz"]) > max_dz or row["distance"] > 1.1 * max_distance:
        return "rejected", "chance alignment"
    if row["group"] == "II":
        return "rejected", "hydrogen-rich"
    if row["group"] not in HYDROGEN_POOR:
        return "rejected", "not a supernova"
    if "band" not in row:
        return "rejected", "no light curve"
    if row["n_detections"] < min_detections:
        return "rejected", "too few detections"
    if row["peak_absolute_magnitude"] <= faint_limit:
        return "control", "brighter than the limit"
    if not row["peak_bracketed"]:
        return "candidate", "peak not bracketed"
    return "candidate", ""


def select(
    classified: pd.DataFrame,
    crossmatch: pd.DataFrame,
    lightcurve_dir: str | Path,
    *,
    Ebv=None,
    max_dz: float = 0.005,
    max_distance: u.Quantity[u.physical.length] = 200 * u.Mpc,
    faint_limit: float = -17.5,
    min_detections: int = 5,
) -> pd.DataFrame:
    """
    Reduce the classified sample to candidates and controls.

    Parameters
    ----------
    classified
        Spectroscopically classified transients, with the columns
        ``source_name``, ``classification``, ``ra``, ``dec`` and ``redshift``.
    crossmatch
        Transient-galaxy pairs, from :func:`read_crossmatch`.
    lightcurve_dir
        Directory of alert light curves named ``lc_<name>.csv``.
    Ebv
        Milky Way reddening of each transient, in the order of ``classified``.
        No extinction is applied if omitted.
    max_dz
        Largest difference in redshift between a transient and its host. The
        default corresponds to 1500 km/s.
    max_distance
        Outer edge of the galaxy catalog.
    faint_limit
        Peak absolute magnitude separating candidates, which are fainter,
        from controls.
    min_detections
        Smallest number of g and r detections.

    Returns
    -------
    :
        One row per transient with its host, its light curve measurements, a
        ``status`` (``"candidate"``, ``"control"`` or ``"rejected"``) and the
        ``reason``. A candidate whose reason is ``"peak not bracketed"`` has a
        peak luminosity that is only a lower limit.

    Notes
    -----
    The cuts are applied in this order: redshift consistent with a matched
    galaxy, hydrogen-poor class, enough detections, peak luminosity.

    Host extinction is not corrected, so every absolute magnitude is a faint
    limit. A red color at peak (g-r > 1) usually points to a reddened normal
    supernova, not to an intrinsically faint one.
    """
    max_distance = max_distance.to_value(u.Mpc)
    matches = dict(tuple(crossmatch.groupby("name")))
    if Ebv is None:
        Ebv = np.zeros(len(classified))

    rows = []
    for (_, source), reddening in zip(classified.iterrows(), Ebv):
        name, redshift = str(source["source_name"]), float(source["redshift"])
        row = {
            "name": name,
            "ra": source["ra"],
            "dec": source["dec"],
            "classification": source["classification"],
            "group": group(source["classification"]),
            "redshift": redshift,
            "Ebv": reddening,
        }
        if redshift > 0:
            row["distance"] = 10 ** (distance_modulus(redshift) / 5 - 5)
        host = associate(matches.get(name, crossmatch.iloc[:0]), redshift)
        if host is not None:
            row |= {
                "host_redshift": host.redshift,
                "dz": host.dz,
                "offset": host.offset,
                "redshift_copied": host.copied,
            }
        lightcurve = read_alerts(Path(lightcurve_dir) / f"lc_{name}.csv")
        if redshift > 0 and not lightcurve.empty:
            row |= _measure(lightcurve, redshift, reddening)
        row["status"], row["reason"] = _label(
            row, max_dz, max_distance, faint_limit, min_detections
        )
        rows.append(row)
    return pd.DataFrame(rows)
