"""Redback prior configuration: redshift mode, kappa, t0 and extinction.

The counterpart of ``nmma_priors.py`` on the other backend, and kept separate
from the fit script for the same reason: what a fit assumes lives in one place
that can be read, tested and diffed without reading the runner.

Order matters and is fixed:

1. ``get_priors(model, constraint=True)``, distributions **and** the
   ``Constraint`` priors bilby needs to reject a sample. Attaching only a
   conversion function (as this pipeline once did) computes the constraint
   ratio and then discards it: bilby rejects a sample only for keys present as
   ``Constraint`` priors.
2. a prior file overrides key by key, never wholesale, so the conversion
   function and the ``Constraint`` entries from step 1 survive.
3. redshift, per the run's mode.
4. kappa, pinned only on request.
5. t0, in MJD, anchored on the first detection.
6. extinction, sampled inside the forward model.
7. the conversion function re-wrapped so it sees the pinned parameters.
8. a check that the constraints really are enforceable.
"""

from __future__ import annotations

import logging

from redback_models import (
    constraint_keys,
    extinction_model,
    has_constraints,
    sampled,
    supports_extinction,
)

# redback's own default for rv_mw in the extinction models. A_V = R_V * E(B-V).
R_V_MW = 3.1

logger = logging.getLogger(__name__)


def _with_fixed_parameters(conversion, priors):
    """Wrap a constraint conversion so it also sees the pinned parameters.

    Some call paths hand the conversion function only a subset of the keys, so
    a DeltaFunction parameter can be missing from the dict. redback's
    constraints read them regardless, ``slsn_constraint`` needs ``kappa`` to
    compute the diffusion timescale, and raise KeyError without this.

    The pinned values are filled in with ``setdefault``, never merged ahead of
    what was passed, for two reasons. A real sampled value must always win over
    the stored peak. And bilby sizes its constraint accumulator from the
    *first* entry of whatever this returns::

        prob = np.ones_like(next(iter(out_sample.values())))

    so putting a scalar peak first makes ``prob`` 0-d, and the subsequent
    ``prob *= constraint`` against a shape-(1,) array raises
    "non-broadcastable output operand". Keeping the caller's keys first keeps
    that accumulator the shape of the sample.
    """
    import bilby

    fixed = {
        name: prior.peak
        for name, prior in priors.items()
        if isinstance(prior, bilby.core.prior.DeltaFunction)
    }

    def wrapped(parameters):
        merged = dict(parameters)
        for name, value in fixed.items():
            merged.setdefault(name, value)
        return conversion(merged)

    return wrapped


def _set_redshift(priors, cfg) -> None:
    """Fixed, photometric, or left exactly as redback ships it.

    The sampled case deliberately does **nothing**: redback's own prior already
    carries a per-model redshift range, and replacing it with a narrower table
    of our own would manufacture an Occam factor against the model with no
    justification for the narrowing.
    """
    import bilby

    shipped = priors["redshift"]

    if cfg.z_known is not None:
        priors["redshift"] = bilby.core.prior.DeltaFunction(
            cfg.z_known, name="redshift", latex_label="$z$"
        )
    elif cfg.z_phot is not None:
        z_min = max(1e-3, float(shipped.minimum))
        priors["redshift"] = bilby.core.prior.TruncatedGaussian(
            mu=cfg.z_phot,
            sigma=cfg.z_phot_err,
            minimum=z_min,
            maximum=float(shipped.maximum),
            name="redshift",
            latex_label="$z$",
        )
        logger.info(
            "redshift ~ TruncatedGaussian(%.4f, %.4f), truncated on redback's "
            "own range [%g, %g]",
            cfg.z_phot,
            cfg.z_phot_err,
            z_min,
            float(shipped.maximum),
        )
    else:
        logger.info(
            "redshift sampled on redback's own prior: %s(%g, %g), left untouched",
            type(shipped).__name__,
            float(shipped.minimum),
            float(shipped.maximum),
        )
    logger.info("redshift: %s", cfg.redshift_mode)


def _set_extinction(priors, cfg) -> None:
    """Put extinction in the forward model, bounded by the dust map.

    ``av_mw`` gets ``Uniform(0, R_V * E(B-V)_map)``: an upper bound from the
    map rather than a fixed value, the same treatment NMMA gives
    ``--Ebv-max``. The map says how much Galactic dust lies along the line of
    sight at most; the data are entitled to prefer less, and marginalising
    lets them. Dereddening the magnitudes instead would impose the map value as
    exact, on SN2021ugl that is E(B-V) = 0.0637 where NMMA's free parameter
    settles at 0.0118, a 0.19 mag over-correction in g.

    ``av_host`` is pinned at 0 unless asked for: there is no dust map to bound
    it with, and on a high-latitude target it is degenerate with the ejecta
    parameters. Sampling it is a scientific choice, not a default.

    Models absent from ``EXTINCTION_MODEL`` get neither key: their t0 wrapper
    is plain ``t0_base_model``, which would reject them.
    """
    import bilby

    if not supports_extinction(cfg.model):
        logger.warning(
            "%s carries NO extinction: its t0 wrapper (%s) mis-scales time, so "
            "neither av_mw nor av_host is sampled. Its ln Z is NOT comparable "
            "with the models that do.",
            cfg.model,
            "t0_tde_extinction",
        )
        return

    av_mw_max = R_V_MW * float(cfg.ebv_mw)
    if av_mw_max > 0:
        priors["av_mw"] = bilby.core.prior.Uniform(
            minimum=0.0,
            maximum=av_mw_max,
            name="av_mw",
            latex_label=r"$A_V^{\mathrm{MW}}$",
        )
        logger.info(
            "av_mw ~ Uniform(0, %.4f) mag, from E(B-V) = %.4f (%s) x R_V = %.1f",
            av_mw_max,
            cfg.ebv_mw,
            cfg.ebv_source,
            R_V_MW,
        )
    else:
        priors["av_mw"] = bilby.core.prior.DeltaFunction(
            0.0, name="av_mw", latex_label=r"$A_V^{\mathrm{MW}}$"
        )
        logger.warning("No E(B-V) bound available: av_mw pinned at 0")

    if cfg.sample_av_host:
        priors["av_host"] = bilby.core.prior.Uniform(
            minimum=0.0,
            maximum=cfg.av_host_max,
            name="av_host",
            latex_label=r"$A_V^{\mathrm{host}}$",
        )
        logger.info("av_host ~ Uniform(0, %.2f) mag", cfg.av_host_max)
    else:
        priors["av_host"] = bilby.core.prior.DeltaFunction(
            0.0, name="av_host", latex_label=r"$A_V^{\mathrm{host}}$"
        )


def build_priors(cfg, reference_mjd: float):
    """The prior actually sampled, with constraints that can reject.

    Parameters
    ----------
    cfg : fit_redback.RunConfig
        A resolved configuration.
    reference_mjd : float
        The first detection, which the t0 bounds are offsets from, the same
        anchoring redback's own examples use.

    Raises
    ------
    ValueError
        If the prior file names a key the model does not have.
    RuntimeError
        If the model should carry constraints but none are enforceable.
    """
    import bilby
    from redback.priors import get_priors

    priors = get_priors(cfg.model, constraint=not cfg.no_constraint)
    if priors is None:
        raise SystemExit(f"redback ships no prior for {cfg.model!r}")
    model_keys = set(sampled(cfg.model))

    if cfg.priors is not None:
        file_priors = bilby.core.prior.PriorDict(str(cfg.priors))
        unknown = set(file_priors) - model_keys
        if unknown:
            raise ValueError(
                f"{cfg.priors} names keys {cfg.model} does not have: "
                f"{sorted(unknown)}. Known keys: {sorted(model_keys)}"
            )
        for key, prior in file_priors.items():
            priors[key] = prior
        logger.info(
            "Prior file: %s (%d key(s) overridden)", cfg.priors, len(file_priors)
        )

    _set_redshift(priors, cfg)

    # Not every model carries an opacity, cooling_envelope and shock_cooling
    # do not, and adding a parameter the model never reads would make bilby
    # sample a dimension with no effect on the likelihood.
    if cfg.kappa is not None:
        if "kappa" in model_keys:
            priors["kappa"] = bilby.core.prior.DeltaFunction(
                cfg.kappa, name="kappa", latex_label=r"$\kappa$"
            )
            logger.info("kappa pinned at %g", cfg.kappa)
        else:
            logger.info("%s has no kappa; --kappa ignored", cfg.model)
    elif "kappa" in model_keys:
        logger.info("kappa sampled")

    # redback ships no t0 prior and says so ("You will need to explicitly set a
    # prior on t0"), so it is ours to set, as an absolute MJD, which is what
    # t0_base_model takes, and anchored on the data exactly as redback's own
    # examples anchor theirs.
    priors["t0"] = bilby.core.prior.Uniform(
        minimum=reference_mjd + cfg.t0_min,
        maximum=reference_mjd + cfg.t0_max,
        name="t0",
        latex_label=r"$t_0$ [MJD]",
    )
    logger.info(
        "t0 in [%.5f, %.5f] MJD (%+g to %+g d from the first detection)",
        priors["t0"].minimum,
        priors["t0"].maximum,
        cfg.t0_min,
        cfg.t0_max,
    )

    _set_extinction(priors, cfg)

    conversion = getattr(priors, "conversion_function", None)
    if cfg.no_constraint:
        for key in constraint_keys(cfg.model):
            priors.pop(key, None)
        priors.conversion_function = bilby.core.prior.PriorDict().conversion_function
        logger.warning("Constraints disabled by --no-constraint")
    elif conversion is not None:
        priors.conversion_function = _with_fixed_parameters(conversion, priors)
        enforceable = [
            key
            for key, prior in priors.items()
            if isinstance(prior, bilby.core.prior.Constraint)
        ]
        if has_constraints(cfg.model) and not enforceable:
            raise RuntimeError(
                f"{cfg.model} has constraints in redback but none are "
                "enforceable in the prior: bilby would compute them and "
                "ignore them. This is the bug the pipeline used to have."
            )
        if enforceable:
            logger.info(
                "Constraint applied: %s via %s",
                ", ".join(sorted(enforceable)),
                getattr(conversion, "__name__", "conversion"),
            )

    return priors


def extinction_summary(cfg) -> dict:
    """How extinction was treated, for the run summary."""
    if not supports_extinction(cfg.model):
        return {
            "treatment": "none",
            "reason": (
                f"{cfg.model} has no usable t0 extinction wrapper "
                "(its spectra time axis is in seconds); ln Z not comparable "
                "with the models that sample extinction"
            ),
        }
    return {
        "treatment": "sampled in the forward model",
        "model": extinction_model(cfg.model),
        "law": "fitzpatrick99",
        "av_mw_max": R_V_MW * float(cfg.ebv_mw),
        "ebv_mw_bound": cfg.ebv_mw,
        "ebv_source": cfg.ebv_source,
        "r_v": R_V_MW,
        "av_host_sampled": bool(cfg.sample_av_host),
    }
