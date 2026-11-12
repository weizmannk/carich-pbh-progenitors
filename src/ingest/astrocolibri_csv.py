"""Convert an Astro-COLIBRI lightcurve CSV into the fitters' ``.dat`` format.

The companion of ``fetch_ztf.py`` for data that arrive as a file rather than
from a broker. Output is the same ``<isot> <band> <mag> <err>`` table, with
``inf`` as the error marking a non-detection.

Usage:
    python astrocolibri_csv.py --object SN2020oi --csv <exported file>

``fetch_ztf.py`` cannot read this CSV, for three reasons worth stating because
each fails quietly rather than loudly:

- ``to_dat_rows`` skips every row whose ``mag`` is empty, which is every upper
  limit: Astro-COLIBRI puts their value in a separate ``mag_ul`` column. The
  limits would simply disappear, and they are what constrain ``t0``.
- the ``flag`` column is ignored, so rows the exporter itself marks ``bad``
  would be fitted alongside the good ones.
- ``filter`` reads ``ztf_g``, which already starts with ``ztf`` and so keeps
  its underscore, a name no bandpass registry resolves.

Instruments are not special-cased: the band name is normalised and then
checked against sncosmo's own registry. ZTF and LSST pass that check,
``asas-sn_g`` does not -- it normalises to ``asas-sng`` -- and the run stops
here instead of hours into a fit, on the first likelihood call.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from carich._paths import DATA

logger = logging.getLogger(__name__)

#: Rows to keep. Everything else is Astro-COLIBRI's own quality verdict and is
#: dropped on its word, never rewritten or re-weighted.
KEPT_FLAGS = ("good", "ul")

#: The flag marking a non-detection, whose magnitude lives in ``mag_ul``.
LIMIT_FLAG = "ul"

#: Error written for a non-detection, the convention the fitters read.
UPPER_LIMIT_ERROR = "inf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", required=True, help="Object name, e.g. SN2020oi")
    parser.add_argument(
        "--csv", required=True, type=Path, help="Astro-COLIBRI lightcurve export"
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Default: data/processed/",
    )
    return parser.parse_args()


def known_band(band: str) -> bool:
    """Whether sncosmo resolves this band name.

    Imported lazily: the check is the only reason this module needs sncosmo,
    and a caller that only wants the parsing should not pay for the import.
    """
    import sncosmo

    try:
        sncosmo.get_bandpass(band)
    except Exception:
        return False
    return True


def to_dat_rows(frame: pd.DataFrame) -> list[tuple[str, str, float, str]]:
    """Normalise an Astro-COLIBRI table into ``.dat`` rows.

    Parameters
    ----------
    frame : pandas.DataFrame
        As exported, with ``time``, ``mjd``, ``filter``, ``mag``, ``mag_err``,
        ``mag_ul`` and ``flag`` columns.

    Returns
    -------
    list of tuple
        ``(isot, band, magnitude, error)``, sorted by MJD, the error kept as a
        string so a non-detection can carry ``inf`` rather than a sentinel.

    Raises
    ------
    SystemExit
        If a band name resolves to nothing sncosmo knows, or if the flag cut
        leaves nothing usable.
    """
    missing = {"time", "mjd", "filter", "mag", "mag_err", "mag_ul", "flag"} - set(
        frame.columns
    )
    if missing:
        raise SystemExit(f"Not an Astro-COLIBRI export: missing {sorted(missing)}")

    kept = frame[frame["flag"].isin(KEPT_FLAGS)].sort_values("mjd").copy()
    kept["band"] = kept["filter"].str.replace("_", "", regex=False)

    unknown = sorted({b for b in kept["band"].unique() if not known_band(b)})
    if unknown:
        source = sorted(kept.loc[kept["band"].isin(unknown), "filter"].unique())
        raise SystemExit(
            f"sncosmo does not know these bands: {unknown}, from {source}. "
            "Map them to a registered name before writing the file, or the "
            "fit will fail on its first likelihood call."
        )

    rows: list[tuple[str, str, float, str]] = []
    for _, row in kept.iterrows():
        limit = row["flag"] == LIMIT_FLAG
        magnitude = row["mag_ul"] if limit else row["mag"]
        if not np.isfinite(float(magnitude)):
            continue
        error = UPPER_LIMIT_ERROR if limit else f"{float(row['mag_err']):.6f}"
        rows.append((str(row["time"]), str(row["band"]), float(magnitude), error))

    if not rows:
        raise SystemExit("No usable row left after the flag cut")
    return rows


def write_dat(rows, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        for isot, band, magnitude, error in rows:
            handle.write(f"{isot} {band} {magnitude:.6f} {error}\n")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    args = parse_args()

    frame = pd.read_csv(args.csv)
    rows = to_dat_rows(frame)

    outdir = args.outdir or DATA / "processed"
    path = outdir / f"{args.object}.dat"
    write_dat(rows, path)

    dropped = len(frame) - sum(1 for _ in rows)
    logger.info("%d of %d rows kept, %d dropped", len(rows), len(frame), dropped)
    for band in sorted({row[1] for row in rows}):
        in_band = [row for row in rows if row[1] == band]
        limits = sum(1 for row in in_band if row[3] == UPPER_LIMIT_ERROR)
        logger.info(
            "  %-10s %d detections, %d limits", band, len(in_band) - limits, limits
        )
    logger.info("written: %s", path)


if __name__ == "__main__":
    main()
