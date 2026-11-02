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
    model_type,
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
            "%s carries NO extinction: redback has no usable t0 wrapper for "
            "its family (%s), so neither av_mw nor av_host is sampled. Its "
            "ln Z is NOT comparable with the models that do.",
            cfg.model,
            model_type(cfg.model),
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


# Priors for the sncosmo source parameters, copied from redback's own
# priors/salt2.prior and from the identical values in
# examples/using_sncosmo_models_to_fit.py. SALT3 takes the same three keys as
# SALT2, so the same values apply.
#
# amplitude is absent: for sn1998bw_template redback priors it as
# LogUniform(1e-2, 1e2), but that is a *normalised* amplitude where 1.0 is the
# original brightness of SN 1998bw, while sncosmo_models passes the raw flux
# scale straight through. No fixed range can be copied, so _amplitude_prior
# derives it through sncosmo's own set_source_peakabsmag.
SNCOSMO_SOURCE_PRIORS = {
    "x0": ("Uniform", dict(minimum=1e-10, maximum=1e-1, latex_label="$x_0$")),
    "x1": ("Normal", dict(mu=0, sigma=1, latex_label="$x_1$")),
    "c": ("Normal", dict(mu=0, sigma=0.1, latex_label="$c$")),
}

# Peak absolute magnitudes the amplitude range is derived to span: a faint
# stripped-envelope event through a superluminous one.
AMPLITUDE_PEAK_ABS_MAG = (-14.0, -22.0)


def _amplitude_prior(template: str, redshift_prior):
    """A LogUniform on the template's raw amplitude, derived, not tabulated.

    The raw ``amplitude`` of a non-SALT source is a flux normalisation, so a
    usable range depends on the distance. Rather than guess one, this asks
    sncosmo what amplitude reproduces a given peak absolute magnitude, through
    the same ``set_source_peakabsmag`` call redback exposes behind
    ``use_set_peak_magnitude``, and brackets the result.

    The bracket spans the **whole** redshift prior, not one value, because the
    span is enormous: for M_B = -19 the v19-1993j-corr amplitude runs from
    9.1e-10 at z = 0.001 to 2.6e-17 at z = 3. Deriving it at a single guess
    left a sampled redshift pressed against the bounds with nothing to say so.

    Parameters
    ----------
    template : str
        An sncosmo source name.
    redshift_prior : bilby.core.prior.Prior
        The redshift prior actually in force. A ``DeltaFunction`` gives one
        value and so the tightest range; a sampled prior gives its full span.

    Returns
    -------
    bilby.core.prior.LogUniform
    """
    import bilby
    import sncosmo
    from astropy.cosmology import Planck18

    peak = getattr(redshift_prior, "peak", None)
    if peak is not None:
        redshifts = [float(peak)]
    else:
        redshifts = [
            max(float(redshift_prior.minimum), 1e-4),
            float(redshift_prior.maximum),
        ]

    model = sncosmo.Model(source=template)
    bounds = []
    for redshift in redshifts:
        for mag in AMPLITUDE_PEAK_ABS_MAG:
            model.set(z=redshift, t0=0.0)
            model.set_source_peakabsmag(
                mag, band="standard::b", magsys="ab", cosmo=Planck18
            )
            bounds.append(float(model.get("amplitude")))
    return bilby.core.prior.LogUniform(
        minimum=min(bounds) / 10.0,
        maximum=max(bounds) * 10.0,
        name="amplitude",
        latex_label="$A$",
    )


def _sncosmo_priors(cfg, reference_mjd: float, bands=None):
    """The prior for redback's sncosmo wrapper, the one model redback leaves open.

    ``get_priors('sncosmo_models')`` returns an empty dict and redback warns
    that "you may need to set up your prior explicitly". That is structural: the
    model is a bridge to the whole sncosmo library and each template has its own
    parameters, so no single file could serve it. redback ships a prior only for
    the templates it promotes to a named model (``salt2``, ``sn1998bw_template``
    and its extrapolated form); everything else, ``salt3`` and ``v19-1993j-corr``
    included, is reached through this generic path and priored by hand.

    This follows ``examples/using_sncosmo_models_to_fit.py``, which is redback's
    own recipe::

        priors['redshift']  = 0.061
        data_peak = sn.x[np.argmin(sn.y)]
        priors['peak_time'] = Uniform(data_peak - 10, data_peak + 10)
        priors['x0'] = Uniform(1e-10, 1e-1)
        priors['x1'] = Normal(0, 1)
        priors['c']  = Normal(0, 0.1)
        kwargs = {..., 'model_kwarg_names': ['x0', 'x1', 'c']}

    with one departure, which is measured rather than assumed: the example also
    samples ``t0``, and ``sncosmo_models`` never reads it. Two calls differing
    only in t0 return identical magnitudes; only ``peak_time`` moves them. A
    sampled t0 would be a flat direction, so it is left out.

    ``ebv`` is the host E(B-V) redback hands to sncosmo's CCM89 effect, the same
    quantity ``--sample-av-host`` controls elsewhere. There is no Galactic
    counterpart: redback adds the F99 effect after ``model.update()`` and never
    sets ``mwebv``, so it stays at 0 and that effect is inert.

    Parameters
    ----------
    cfg : fit_redback.RunConfig
        A resolved configuration; ``cfg.sncosmo_model`` picks the template.
    reference_mjd : float
        The first detection.
    bands : sequence of str, optional
        The bands being fitted, used to bound the redshift prior at the
        template's spectral validity. None leaves that prior untouched.
    """
    import bilby
    from redback_models import (
        sncosmo_keys,
        sncosmo_source_parameters,
    )

    template = cfg.sncosmo_model
    priors = bilby.core.prior.PriorDict()
    # 0.01 is the lower bound of redback's own sn1998bw_template.prior, 3 the
    # upper bound it gives its supernova models. Overridden by the versioned
    # prior file, and replaced by --z-known / --z-phot when either is given.
    priors["redshift"] = bilby.core.prior.Uniform(
        minimum=0.01, maximum=3, name="redshift", latex_label="$z$"
    )

    for key in sncosmo_source_parameters(template):
        if key == "amplitude":
            continue  # set below, once the redshift prior is settled
        if key not in SNCOSMO_SOURCE_PRIORS:
            raise ValueError(
                f"{template} has source parameter {key!r}, which no redback "
                "prior covers. Add it to SNCOSMO_SOURCE_PRIORS with the redback "
                "file it is copied from, or choose another template."
            )
        kind, settings = SNCOSMO_SOURCE_PRIORS[key]
        priors[key] = getattr(bilby.core.prior, kind)(name=key, **settings)
    logger.info("%s samples: %s", template, ", ".join(sncosmo_keys(template)))

    if cfg.priors is not None:
        known = set(sncosmo_keys(template))
        file_priors = bilby.core.prior.PriorDict(str(cfg.priors))
        unknown = set(file_priors) - known
        if unknown:
            raise ValueError(
                f"{cfg.priors} names keys {template} does not have: "
                f"{sorted(unknown)}. Known keys: {sorted(known)}"
            )
        for key, prior in file_priors.items():
            priors[key] = prior
        logger.info(
            "Prior file: %s (%d key(s) overridden)", cfg.priors, len(file_priors)
        )

    # The redshift prior is left exactly as declared. Past the redshift where a
    # bandpass leaves the template's wavelength range, sncosmo raises; that is
    # handled in redback_models._reject_outside_template, which returns an
    # infinite magnitude so the likelihood rejects the sample. Truncating the
    # prior instead would remove volume and inflate ln Z by the log of the
    # width removed, for a reason unrelated to the data.

    _set_redshift(priors, cfg)

    # After _set_redshift, so the bracket spans the redshift prior actually in
    # force: pinned by --z-known, truncated by --z-phot, or redback's full range.
    # Only when nothing supplied it. A --priors file must win over a derived
    # default, as it does for every other key: deriving on top would make the
    # file's amplitude line dead, which it silently was.
    if "amplitude" in sncosmo_source_parameters(template) and "amplitude" not in priors:
        priors["amplitude"] = _amplitude_prior(template, priors["redshift"])
        logger.info(
            "amplitude ~ LogUniform(%.3e, %.3e), derived for M_peak in %s over "
            "the redshift prior in force",
            priors["amplitude"].minimum,
            priors["amplitude"].maximum,
            AMPLITUDE_PEAK_ABS_MAG,
        )

    if cfg.sample_av_host:
        ebv_max = cfg.av_host_max / R_V_MW
        priors["ebv"] = bilby.core.prior.Uniform(
            minimum=0.0,
            maximum=ebv_max,
            name="ebv",
            latex_label=r"$E(B-V)_{\mathrm{host}}$",
        )
        logger.info(
            "ebv ~ Uniform(0, %.4f) mag, host only, from av_host_max = %.2f / "
            "R_V = %.1f",
            ebv_max,
            cfg.av_host_max,
            R_V_MW,
        )
    else:
        priors["ebv"] = bilby.core.prior.DeltaFunction(
            0.0, name="ebv", latex_label=r"$E(B-V)_{\mathrm{host}}$"
        )

    # A fixed offset window on the first epoch, exactly like T0_WINDOW for the
    # other models, and bounded by the same --t0-min / --t0-max flags, which
    # this path otherwise leaves unused because the model never reads t0.
    #
    # NOT centred on the brightest detection. redback's example sets this as
    # Uniform(data_peak - 10, data_peak + 10) with
    # data_peak = sn.x[np.argmin(sn.y)], which is right for the model it fits,
    # salt2, whose light curve has one peak: the brightest point is then the
    # template's maximum by construction. A double-peaked light curve breaks
    # that. On SN2021ugl, a Type IIb, the brightest detection is the Ni-56 peak
    # 18 days after the shock-cooling one, so centring there puts phase zero at
    # +18 d and leaves the first 19 days outside the template's coverage, where
    # sncosmo clamps to the edge value. Measured over the full 47 days, chi2 per
    # detection: 4.1 with phase zero at the first peak, 127.3 at the brightest
    # point. NMMA fits the same template over the same data because it never
    # sets the sncosmo t0, leaving phase zero on the trigger.
    priors["peak_time"] = bilby.core.prior.Uniform(
        minimum=reference_mjd + cfg.t0_min,
        maximum=reference_mjd + cfg.t0_max,
        name="peak_time",
        latex_label=r"$t_{\mathrm{peak}}$ [MJD]",
    )
    logger.info(
        "peak_time in [%.5f, %.5f] MJD (%+g to %+g d from the first epoch)",
        priors["peak_time"].minimum,
        priors["peak_time"].maximum,
        cfg.t0_min,
        cfg.t0_max,
    )
    logger.warning(
        "Galactic extinction is NOT applied on the sncosmo path: redback adds "
        "the F99 effect but never sets mwebv, so it stays at 0. ln Z is not "
        "comparable with the models that sample av_mw."
    )
    if cfg.ebv_mw:
        logger.warning(
            "--ebv-mw %.4f is IGNORED here. It bounds av_mw, which only the "
            "physical models carry; this path has no Galactic term at all. The "
            "host E(B-V) it might look like is the separate 'ebv' key, driven "
            "by --sample-av-host.",
            cfg.ebv_mw,
        )
    if not cfg.sample_av_host:
        logger.info(
            "ebv pinned at 0: host extinction is off unless --sample-av-host "
            "is given. Note redback's own default for this key is 0.1, which "
            "would redden every template silently; 0 is set deliberately."
        )

    return priors


def build_priors(cfg, reference_mjd: float, bands=None):
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
    from redback_models import SNCOSMO_MODEL

    if cfg.model == SNCOSMO_MODEL:
        return _sncosmo_priors(cfg, reference_mjd, bands)

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
    from redback_models import SNCOSMO_MODEL

    if cfg.model == SNCOSMO_MODEL:
        return {
            "treatment": "host only, inside sncosmo",
            "law": "ccm89 (host, rest frame)",
            "ebv_host_sampled": bool(cfg.sample_av_host),
            "ebv_host_max": (cfg.av_host_max / R_V_MW if cfg.sample_av_host else 0.0),
            "reason": (
                "redback adds sncosmo's F99 Milky Way effect but never sets "
                "mwebv, so Galactic extinction is inert on this path; ln Z "
                "not comparable with the models that sample av_mw"
            ),
        }
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
