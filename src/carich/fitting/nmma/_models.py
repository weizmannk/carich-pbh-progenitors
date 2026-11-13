"""NMMA model -> --em-transient-class lookup table, and prior file resolution.

Without an explicit --em-transient-class, nmma falls back to an SVD class
and tries to download a model that doesn't exist. This table turns that
into an immediate, clear failure instead of a confusing remote 404.
"""

from __future__ import annotations

import logging
from pathlib import Path

_SNCOSMO_GENERIC: set[str] = {
    "nugent-sn1a",
    "nugent-sn1bc",
    "nugent-hyper",
    "nugent-sn2p",
    "nugent-sn2l",
    "nugent-sn2n",
    "s11-2005hl",
    "s11-2006jo",
    "snana-2004fe",
    "v19-2016gkg-corr",
    "v19-2011dh-corr",
    "v19-2016gkg",
    "v19-2011dh",
    "v19-1993j-corr",
    "v19-1994i-corr",
    "v19-2012au-corr",
    "v19-2008d-corr",
}

_COMBINED_PRIORS: dict[frozenset, str] = {
    frozenset({"nugent-hyper", "sr2023"}): "Sr2023.prior",
    frozenset({"nugent-hyper", "trpi2018"}): "sncosmo-generic_TrPi2018.prior",
}

# Only models whose class can't be inferred from the name live here. SN
# templates (the only ones relevant to this project: SALT3/Nugent/v19-*)
# are detected via the sncosmo registry, not listed explicitly.
_MODEL_TO_CLASS: dict[str, str | None] = {
    "trpi2018": "grb",
    "piro2021": "shock",
    "me2017": "simple_kilonova",
    "hona2020": "simple_kilonova",
    "pl_bb_fixedt": "simple_kilonova",
    "blackbody_fixedt": "simple_kilonova",
    "synchrotron_powerlaw": "simple_kilonova",
    "sr2023": "host_galaxy",
    "arnett": None,
    "arnett_modified": None,
}


def _is_sncosmo_template(model: str) -> bool:
    """Check the model against the sncosmo registry (authoritative source)."""
    try:
        from sncosmo.models import _SOURCES

        return model in {v["name"] for v in _SOURCES.get_loaders_metadata()}
    except Exception:
        return False


def infer_transient_class(model: str) -> str | None:
    """Infer --em-transient-class from model name(s) (comma-separated).

    Any sncosmo template (registry, including salt2/salt3) -> "supernova";
    unknown name -> explicit "svd". Returns None if a component has no CLI
    class (the flag must then be omitted).
    """
    classes = []
    for part in model.split(","):
        m = part.strip().lower()
        if m in _MODEL_TO_CLASS:
            cls = _MODEL_TO_CLASS[m]
            if cls is None:
                logging.info(
                    "infer_transient_class: %r has no CLI class; --em-transient-class omitted.",
                    part,
                )
                return None
        elif (
            m in _SNCOSMO_GENERIC
            or _is_sncosmo_template(part.strip())
            or m.startswith(("salt", "nugent", "snana", "v19", "s11"))
        ):
            cls = "supernova"
        else:
            logging.warning(
                "infer_transient_class: unknown model %r, defaulting to 'svd'", part
            )
            cls = "svd"
        classes.append(cls)
    return ",".join(classes)


def resolve_prior_file(model: str, priors_dir: Path) -> Path:
    """Resolve the prior file for a single or combined model (comma-separated).

    Piro2021 + any sncosmo-generic model combinations share a common prior;
    otherwise fall back to the explicit table (_COMBINED_PRIORS).
    """
    priors_dir = Path(priors_dir)
    m = model.lower().strip()

    if "," in m:
        parts = frozenset(p.strip() for p in m.split(","))
        if "piro2021" in parts and (parts - {"piro2021"}) <= _SNCOSMO_GENERIC:
            return priors_dir / "sncosmo-generic_Piro2021.prior"
        if parts in _COMBINED_PRIORS:
            return priors_dir / _COMBINED_PRIORS[parts]
        combined = "_".join(sorted(parts))
        logging.warning(
            "resolve_prior_file: unknown combination %r, trying %s.prior",
            model,
            combined,
        )
        return priors_dir / f"{combined}.prior"

    if m in _SNCOSMO_GENERIC:
        return priors_dir / "sncosmo-generic.prior"
    if ("salt2" in m) or ("salt3" in m):
        return priors_dir / "salt3.prior"
    if m == "piro2021":
        return priors_dir / "Piro2021.prior"
    return priors_dir / f"{model}.prior"
