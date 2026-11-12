from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HostMatch:
    """
    Galaxy of the Census of the Local Universe matched to a transient.
    """

    clu_id: int
    """Identifier of the galaxy in the CLU catalog."""

    redshift: float
    """Redshift of the galaxy."""

    dz: float
    """Redshift of the transient minus redshift of the galaxy."""

    offset: float
    """Projected separation at the distance of the galaxy, in kpc."""

    separation: float
    """Angular separation, in arcsec."""

    @property
    def copied(self) -> bool:
        """
        Whether the two redshifts are identical.

        A transient whose redshift equals that of the galaxy to five decimal
        places was most likely assigned the host redshift. The match then
        confirms nothing until the redshift is measured in the spectrum of
        the transient itself.
        """
        return abs(self.dz) < 2e-5


def read_crossmatch(path: str | Path) -> pd.DataFrame:
    """
    Read the table of transient-galaxy pairs.

    Parameters
    ----------
    path
        CSV file with one row per pair and the columns ``name``, ``clu_id``,
        ``clu_z``, ``clu_dist_kpc`` and ``clu_sep_arcsec``. A transient
        appears once for each galaxy it was matched to.

    Returns
    -------
    :
        The same table with stripped column names and numeric columns.
    """
    table = pd.read_csv(path, skipinitialspace=True)
    table.columns = [column.strip() for column in table.columns]
    table["name"] = table["name"].astype(str).str.strip()
    for column in ("clu_z", "clu_dist_kpc", "clu_sep_arcsec"):
        table[column] = pd.to_numeric(table[column], errors="coerce")
    return table


def associate(matches: pd.DataFrame, redshift: float) -> HostMatch | None:
    """
    Pick the matched galaxy closest in redshift to a transient.

    Parameters
    ----------
    matches
        Rows of the crossmatch table for one transient.
    redshift
        Redshift of the transient.

    Returns
    -------
    :
        The best match, or `None` if the transient has no redshift or no
        matched galaxy with a redshift.

    Notes
    -----
    The crossmatch pairs a transient with every galaxy within 10-130 kpc in
    projection, whatever its redshift. A pair with a large ``dz`` is a chance
    alignment: the galaxy is not the host and the projected offset has no
    meaning. Whether ``dz`` is small enough is left to the caller.
    """
    if not np.isfinite(redshift):
        return None
    matches = matches.dropna(subset=["clu_z"])
    if matches.empty:
        return None
    row = matches.iloc[np.abs(matches["clu_z"].to_numpy() - redshift).argmin()]
    return HostMatch(
        clu_id=int(row["clu_id"]),
        redshift=float(row["clu_z"]),
        dz=float(redshift - row["clu_z"]),
        offset=float(row["clu_dist_kpc"]),
        separation=float(row["clu_sep_arcsec"]),
    )
