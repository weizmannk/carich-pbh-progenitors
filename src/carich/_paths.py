"""Locations of the repository files that the package reads."""

from pathlib import Path


def root() -> Path:
    """
    Root of the repository.

    Notes
    -----
    The data and the prior files sit next to ``src``, not inside the package,
    so the result is valid only for an editable installation, the one that
    the ``Makefile`` sets up.
    """
    return Path(__file__).resolve().parents[2]


DATA = root() / "data"
"""Catalogs, models and photometry."""

PRIORS_REDBACK = root() / "priors_redback"
"""Prior files of the redback models, named ``<model>.prior``."""

PRIORS_NMMA = root() / "priors_nmma"
"""Prior files of the NMMA models."""
