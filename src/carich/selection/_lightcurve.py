from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Peak:
    """
    Brightest detection of a light curve in one band.
    """

    time: float
    """Epoch of the brightest detection, in days."""

    magnitude: float
    """Apparent magnitude of the brightest detection."""

    error: float
    """Uncertainty on the magnitude."""

    bracketed: bool
    """Whether the light curve is seen both rising to and falling from it."""


def read_alerts(path: str | Path, max_error: float = 0.25) -> pd.DataFrame:
    """
    Read a ZTF alert light curve.

    Parameters
    ----------
    path
        CSV file of alert packets with the columns ``jd``, ``filter``,
        ``magpsf``, ``sigmapsf`` and ``isdiffpos``.
    max_error
        Largest magnitude uncertainty kept.

    Returns
    -------
    :
        Positive-subtraction detections in g, r and i, sorted by time. The
        table is empty if the file does not exist.

    Notes
    -----
    Alert packets carry detections only. Without upper limits a light curve
    that starts at its brightest point cannot be told from one caught on the
    decline; use forced photometry before quoting a peak luminosity.
    """
    columns = ["jd", "filter", "magpsf", "sigmapsf"]
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=columns)
    table = pd.read_csv(path)
    positive = table["isdiffpos"].astype(str).str.lower().isin(("t", "1", "true"))
    keep = (
        positive & (table["sigmapsf"] < max_error) & table["filter"].isin(list("gri"))
    )
    return table.loc[keep, columns].sort_values("jd", ignore_index=True)


def peak(time, magnitude, error) -> Peak | None:
    """
    Find the brightest detection and test whether it is a maximum.

    Parameters
    ----------
    time
        Epochs of the detections in one band, in days.
    magnitude
        Apparent magnitudes.
    error
        Uncertainties on the magnitudes.

    Returns
    -------
    :
        The brightest detection, or `None` if there is no detection.

    Notes
    -----
    The peak is bracketed if a detection at least one day earlier and another
    at least one day later are both significantly fainter: by 0.15 mag and by
    twice the combined uncertainty. Otherwise the brightest point may be the
    first or last epoch of a monotonic light curve, and the true maximum is
    brighter.
    """
    time, magnitude, error = map(np.asarray, (time, magnitude, error))
    if time.size == 0:
        return None
    i = magnitude.argmin()
    fainter = magnitude > magnitude[i] + np.maximum(0.15, 2 * np.hypot(error, error[i]))
    before = fainter & (time < time[i] - 1)
    after = fainter & (time > time[i] + 1)
    return Peak(
        time=float(time[i]),
        magnitude=float(magnitude[i]),
        error=float(error[i]),
        bracketed=bool(before.any() and after.any()),
    )


def interpolate(time, magnitude, epoch: float, max_gap: float) -> float:
    """
    Magnitude at an epoch, by linear interpolation between detections.

    Parameters
    ----------
    time
        Epochs of the detections in one band, in days, sorted.
    magnitude
        Apparent magnitudes.
    epoch
        Epoch at which to evaluate the light curve.
    max_gap
        Longest interval between the two neighbouring detections, in days,
        across which to interpolate.

    Returns
    -------
    :
        Interpolated magnitude. If the gap is too long, the nearest detection
        is used when it lies within two days of the epoch. NaN otherwise, or
        if the epoch falls outside the light curve.
    """
    time, magnitude = np.asarray(time), np.asarray(magnitude)
    if time.size < 2 or not time[0] <= epoch <= time[-1]:
        return np.nan
    j = np.clip(np.searchsorted(time, epoch), 1, time.size - 1)
    if time[j] - time[j - 1] <= max_gap:
        return float(np.interp(epoch, time, magnitude))
    nearest = np.abs(time - epoch).argmin()
    if abs(time[nearest] - epoch) <= 2:
        return float(magnitude[nearest])
    return np.nan
