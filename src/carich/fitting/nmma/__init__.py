"""Template fits with NMMA, as a comparison to the redback fits.

The fit, its figure and the Condor argument builder are run as modules::

    python -m carich.fitting.nmma._fit --help
    python -m carich.fitting.nmma._plot --help
    python -m carich.fitting.nmma._condor --help
"""

# _fit, _plot and _condor are left out: importing them here would load them
# twice under ``python -m``.
from ._models import infer_transient_class, resolve_prior_file
from ._priors import configure_prior, create_prior

__all__ = (
    "configure_prior",
    "create_prior",
    "infer_transient_class",
    "resolve_prior_file",
)
