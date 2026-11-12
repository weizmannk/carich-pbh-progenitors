"""Fetch ZTF photometry for one object and write it in the fitters' format.

Pulls alert photometry from a ZTF broker -- Fink by default, Lasair as the
alternative -- through redback's own getters, so the repository depends on no
private service. The output is the ``<isot> <band> <mag> <err>`` file that
``src/fitting/*`` already read, with ``inf`` as the error marking a
non-detection.

Usage:
    python fetch_ztf.py --object ZTF20abaunmw
    python fetch_ztf.py --object ZTF20abaunmw --broker lasair --tmax 47

Brokers carry alert photometry, which is not forced photometry: non-detections
are reported only where the broker chose to, and the earliest epochs can be
missing entirely. For a measurement rather than a pipeline test, request ZTF
forced photometry and convert it with ``--from-csv``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.time import Time

from carich._paths import DATA

logger = logging.getLogger(__name__)

#: Broker filter id -> the band name redback's filter table expects.
ZTF_FILTERS = {1: "ztfg", 2: "ztfr", 3: "ztfi"}

#: Error written for a non-detection. The fitters read `inf` as "this is an
#: upper limit, the magnitude is the limiting magnitude".
UPPER_LIMIT_ERROR = "inf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", required=True, help="ZTF name, e.g. ZTF20abaunmw")
    parser.add_argument(
        "--broker",
        default="fink",
        choices=("fink", "lasair"),
        help="Which ZTF broker to query",
    )
    parser.add_argument(
        "--transient-type",
        default="supernova",
        dest="transient_type",
        help="Transient class the broker getter expects",
    )
    parser.add_argument(
        "--from-csv",
        type=Path,
        default=None,
        dest="from_csv",
        help=(
            "Convert a local CSV instead of querying a broker. Expects "
            "columns mjd, filter, mag, magerr (ZTF forced-photometry style)."
        ),
    )
    parser.add_argument(
        "--tmax",
        type=float,
        default=None,
        help="Keep only epochs within this many days of the first detection",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Default: data/processed/ at the repository root",
    )
    return parser.parse_args()


def fetch_from_broker(name: str, broker: str, transient_type: str) -> pd.DataFrame:
    """Query a ZTF broker through redback.

    Parameters
    ----------
    name : str
        ZTF object name.
    broker : {'fink', 'lasair'}
        Which broker to query.
    transient_type : str
        Class the getter expects, e.g. ``supernova``.

    Returns
    -------
    pandas.DataFrame
        Whatever the broker returned, unmodified.
    """
    from redback.get_data import get_fink_data, get_lasair_data

    getter = get_fink_data if broker == "fink" else get_lasair_data
    logger.info("Querying %s for %s", broker, name)
    return getter(transient=name, transient_type=transient_type)


def read_forced_photometry(path: Path) -> pd.DataFrame:
    """Read a local ZTF forced-photometry CSV.

    Accepts the column spellings the ZTF products use interchangeably
    (``mjd``/``jd``, ``filter``/``fid``, ``magerr``/``sigmamag``).
    """
    frame = pd.read_csv(path)
    frame.columns = [column.strip().lower() for column in frame.columns]
    if "mjd" not in frame and "jd" in frame:
        frame["mjd"] = frame["jd"] - 2400000.5
    return frame


def to_dat_rows(frame: pd.DataFrame) -> list[tuple[float, str, float, str]]:
    """Normalise a broker or forced-photometry table into ``.dat`` rows.

    Returns
    -------
    list of tuple
        ``(mjd, band, magnitude, error)`` with the error as a string so a
        non-detection can carry ``inf`` rather than a sentinel number.
    """
    columns = set(frame.columns)

    mjd_column = next((c for c in ("mjd", "time", "jd") if c in columns), None)
    mag_column = next((c for c in ("magpsf", "mag", "magnitude") if c in columns), None)
    err_column = next(
        (
            c
            for c in ("sigmapsf", "magerr", "sigmamag", "magnitude_err")
            if c in columns
        ),
        None,
    )
    band_column = next((c for c in ("fid", "filter", "band") if c in columns), None)

    if mjd_column is None or mag_column is None or band_column is None:
        raise SystemExit(
            f"Cannot map columns {sorted(columns)} onto mjd/mag/band. "
            "Pass --from-csv with a ZTF forced-photometry file instead."
        )

    rows: list[tuple[float, str, float, str]] = []
    for _, row in frame.iterrows():
        band_value = row[band_column]
        band = (
            ZTF_FILTERS.get(int(band_value))
            if str(band_value).isdigit()
            else str(band_value).strip().lower()
        )
        if band is None:
            continue
        if not band.startswith("ztf"):
            band = f"ztf{band}" if band in ("g", "r", "i") else band

        magnitude = row[mag_column]
        if pd.isna(magnitude):
            continue

        error = row[err_column] if err_column else np.nan
        error_text = (
            UPPER_LIMIT_ERROR
            if pd.isna(error) or float(error) <= 0
            else f"{float(error)}"
        )
        rows.append((float(row[mjd_column]), band, float(magnitude), error_text))

    if not rows:
        raise SystemExit("Broker returned no usable photometry")
    rows.sort(key=lambda entry: entry[0])
    return rows


def apply_window(rows, tmax: float | None):
    """Keep epochs within ``tmax`` days of the first detection."""
    if tmax is None:
        return rows
    first_detection = next(
        (mjd for mjd, _, _, err in rows if err != UPPER_LIMIT_ERROR), None
    )
    if first_detection is None:
        return rows
    kept = [row for row in rows if row[0] - first_detection <= tmax]
    logger.info("Window %.0f d: %d of %d epochs kept", tmax, len(kept), len(rows))
    return kept


def write_dat(rows, path: Path) -> None:
    """Write the ``<isot> <band> <mag> <err>`` file the fitters read."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        for mjd, band, magnitude, error in rows:
            isot = Time(mjd, format="mjd").isot
            handle.write(f"{isot} {band} {magnitude} {error}\n")


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    if args.from_csv is not None:
        frame = read_forced_photometry(args.from_csv)
    else:
        frame = fetch_from_broker(args.object, args.broker, args.transient_type)

    rows = apply_window(to_dat_rows(frame), args.tmax)
    outdir = args.outdir or DATA / "processed"
    path = outdir / f"{args.object}.dat"
    write_dat(rows, path)

    detections = sum(1 for row in rows if row[3] != UPPER_LIMIT_ERROR)
    bands = sorted({row[1] for row in rows})
    logger.info(
        "%d epochs (%d detections, %d limits) in %s",
        len(rows),
        detections,
        len(rows) - detections,
        ", ".join(bands),
    )
    logger.info("written: %s", path)


if __name__ == "__main__":
    main()
