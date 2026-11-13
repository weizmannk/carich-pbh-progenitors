import io
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.time import Time

URL = "https://fallingstar-data.com/forcedphot"
"""Address of the ATLAS forced photometry server."""

BANDS = {"c": "atlasc", "o": "atlaso"}
"""ATLAS filters and their :mod:`sncosmo` names."""

_ZEROPOINT = 23.9
"""AB magnitude of a source of 1 microjansky."""


def request(
    ra: float,
    dec: float,
    mjd_min: float,
    mjd_max: float,
    token: str | None = None,
    poll: float = 10,
) -> str:
    """
    Request difference-image forced photometry at a sky position.

    Parameters
    ----------
    ra, dec
        ICRS coordinates in degrees.
    mjd_min, mjd_max
        Range of dates.
    token
        Token of an account on the server. Read from the environment
        variable ``ATLAS_TOKEN`` if omitted.
    poll
        Interval between two checks of the job, in seconds.

    Returns
    -------
    :
        Photometry table as text, one row per exposure, to pass to
        :func:`read`.

    Notes
    -----
    The job is queued on the server and usually completes within minutes.
    Registration is free at https://fallingstar-data.com/forcedphot/.
    """
    import requests

    token = token or os.environ["ATLAS_TOKEN"]
    headers = {"Authorization": f"Token {token}", "Accept": "application/json"}
    data = {
        "ra": ra,
        "dec": dec,
        "mjd_min": mjd_min,
        "mjd_max": mjd_max,
        "use_reduced": False,
        "send_email": False,
    }

    with requests.Session() as session:
        while True:
            response = session.post(f"{URL}/queue/", headers=headers, data=data)
            if response.status_code != 429:  # not throttled
                break
            time.sleep(poll)
        response.raise_for_status()
        task = response.json()["url"]

        while True:
            status = session.get(task, headers=headers)
            status.raise_for_status()
            status = status.json()
            if status.get("error_msg"):
                raise RuntimeError(status["error_msg"])
            if status.get("finishtimestamp"):
                break
            time.sleep(poll)

        result = session.get(status["result_url"], headers=headers)
        result.raise_for_status()
        return result.text


def read(text: str) -> pd.DataFrame:
    """
    Parse a table returned by the forced photometry server.

    Parameters
    ----------
    text
        Table as text, with the header line starting with ``###MJD``.

    Returns
    -------
    :
        One row per exposure. Fluxes ``uJy`` and their uncertainties ``duJy``
        are in microjansky; ``F`` is the filter, ``c`` or ``o``.
    """
    table = pd.read_csv(io.StringIO(text.replace("###", "")), sep=r"\s+")
    return table.astype({"MJD": float, "uJy": float, "duJy": float})


def clean(
    table: pd.DataFrame, max_error: float = 160, max_chi2: float = 10
) -> pd.DataFrame:
    """
    Drop exposures of poor quality.

    Parameters
    ----------
    table
        Exposures, from :func:`read`.
    max_error
        Largest flux uncertainty kept, in microjansky.
    max_chi2
        Largest reduced chi-squared of the PSF fit kept.
    """
    good = (table["duJy"] > 0) & (table["duJy"] < max_error)
    good &= table["chi/N"] < max_chi2
    return table[good].reset_index(drop=True)


def stack(table: pd.DataFrame, snr: float = 3, max_gap: float = 0.4) -> pd.DataFrame:
    """
    Combine the exposures of each night and convert to magnitudes.

    Parameters
    ----------
    table
        Exposures, from :func:`clean`.
    snr
        Signal-to-noise ratio above which a nightly flux is a detection.
    max_gap
        Longest interval, in days, between consecutive exposures of the same
        night in the same filter.

    Returns
    -------
    :
        One row per night and filter with the columns ``mjd``, ``band``,
        ``mag`` and ``magerr``. A non-detection carries the limiting
        magnitude at the same signal-to-noise ratio and an infinite
        uncertainty, the convention of the fitters.

    Notes
    -----
    ATLAS takes about four 30 s exposures of a field per night. The nightly
    flux is their inverse-variance weighted mean, which gains about 0.75 mag
    in depth over a single exposure.
    """
    rows = []
    for band, exposures in table.sort_values("MJD").groupby("F"):
        night = (exposures["MJD"].diff() > max_gap).cumsum()
        for _, group in exposures.groupby(night):
            weight = group["duJy"] ** -2
            flux = np.sum(weight * group["uJy"]) / weight.sum()
            error = weight.sum() ** -0.5
            detected = flux > snr * error
            rows.append(
                {
                    "mjd": np.sum(weight * group["MJD"]) / weight.sum(),
                    "band": BANDS[band],
                    "mag": _ZEROPOINT
                    - 2.5 * np.log10(flux if detected else snr * error),
                    "magerr": 2.5 / np.log(10) * error / flux if detected else np.inf,
                }
            )
    stacked = pd.DataFrame(rows, columns=["mjd", "band", "mag", "magerr"])
    return stacked.sort_values("mjd", ignore_index=True)


def write(table: pd.DataFrame, path: str | Path, append: bool = False) -> None:
    """
    Write photometry in the format read by the fitters.

    Parameters
    ----------
    table
        Photometry with the columns ``mjd``, ``band``, ``mag`` and ``magerr``.
    path
        Output file, one ``<isot> <band> <mag> <err>`` line per epoch.
    append
        Merge with the epochs already in the file, so that ATLAS can be
        added to an existing ZTF light curve.
    """
    path = Path(path)
    lines = [
        f"{Time(row.mjd, format='mjd').isot} {row.band} {row.mag:.4f} {row.magerr:.4g}"
        for row in table.itertuples()
    ]
    if append and path.exists():
        lines += path.read_text().splitlines()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(sorted(set(lines))) + "\n")
