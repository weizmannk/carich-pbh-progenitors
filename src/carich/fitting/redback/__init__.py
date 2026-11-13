"""Fits of the redback models, with Galactic extinction sampled in the model.

The fit and the Condor argument builder are run as modules::

    python -m carich.fitting.redback._fit --help
    python -m carich.fitting.redback._condor --help
"""

# _fit and _condor are left out: importing them here would load them twice
# under ``python -m``. _plot is left out because it selects the Agg backend
# of matplotlib when imported.
from ._data import (
    build_transient,
    first_detection_mjd,
    first_epoch_mjd,
    read_photometry,
)
from ._descriptions import DESCRIPTIONS, describe
from ._models import (
    DEFAULT_LIKELIHOOD,
    LIKELIHOODS,
    MODELS,
    SNCOSMO_MODEL,
    check_against_redback,
    likelihood_class,
    model_kwargs,
    observed_model,
    sncosmo_prior_stem,
    t0_window,
    time_parameter,
)
from ._priors import build_priors, extinction_summary

__all__ = (
    "DEFAULT_LIKELIHOOD",
    "DESCRIPTIONS",
    "LIKELIHOODS",
    "MODELS",
    "SNCOSMO_MODEL",
    "build_priors",
    "build_transient",
    "check_against_redback",
    "describe",
    "extinction_summary",
    "first_detection_mjd",
    "first_epoch_mjd",
    "likelihood_class",
    "model_kwargs",
    "observed_model",
    "read_photometry",
    "sncosmo_prior_stem",
    "t0_window",
    "time_parameter",
)
