"""Which redback models this project fits, and the one input redback needs.

Everything here is read from redback or required by it. Nothing is substituted,
guarded or re-derived:

- the sampled parameter list comes from ``redback.priors.get_priors``;
- the extinction wrapper comes from redback's own ``model_type`` metadata;
- the constraints come from ``get_priors(model, constraint=True)``;
- the model output is passed through untouched.

Only :data:`T0_WINDOW` is supplied here, because redback ships no t0 prior and
says so itself (``priors.py``: "You will need to explicitly set a prior on t0").

:data:`MODELS` is a project allowlist, not a redback concept: redback exposes
412 models and this analysis uses five.
"""

from __future__ import annotations

import numpy as np

# The five models this project fits. Used as the --model choices.
MODELS: tuple[str, ...] = (
    "arnett",
    "slsn",
    "shock_cooling",
    "tde_analytical",
    "cooling_envelope",
    # Both peaks of a Type IIb in one model: the union of shock_cooling's and
    # arnett's parameters. redback ships no metadata for it either, so its
    # family is in the fallback below, verified: it lives in
    # supernova_models and t0_supernova_extinction reproduces t0_base_model to
    # 0.000000 mag.
    "shock_cooling_and_arnett",
    # redback's general interface to every sncosmo template, so the templates
    # NMMA fits (v19-1993j-corr, salt3, ...) are reachable here too. It reaches
    # t0_base_model like the rest; what differs is that its prior is not
    # shipped. See SNCOSMO_KEYS.
    "sncosmo_models",
)

# redback's key for the sncosmo wrapper, the one model here that is not a
# redback forward model but a bridge to sncosmo's template library.
SNCOSMO_MODEL = "sncosmo_models"

# Bounds on the time origin, in days relative to the first detection. redback
# samples t0 as an absolute MJD, t0_base_model runs it through
# ``Time(t0, format='mjd')``, so a bare offset would be read as an epoch in
# 1858, and both of redback's own examples derive the bounds from the data the
# same way:
#
#     priors['t0'] = Uniform(data['MJD'].values.min() - 200,
#                            data['MJD'].values.min() - 1, name='t0')
#     priors['t0'] = Uniform(supernova.x[0] - 10, supernova.x[0] - 0.05, name='t0')
#
# These are the offsets _priors adds to the first detection. redback
# cannot ship them: the plausible range depends on the data, not on the model.
# Override with --t0-min / --t0-max.
#
# The supernova windows follow the NMMA Astro-COLIBRI priors, which put
# timeshift in Uniform(-2, +1) day of the trigger. The first Astro-COLIBRI
# detection pins the epoch to about a day, so a window of tens of days only
# lets the sampler wander. The upper bound is positive, as NMMA's is: t0 is
# allowed just past the first detection and the likelihood, not the bound,
# rules it out (t0_base_model returns its faint sentinel before t0, so such a
# sample scores badly on the first detection and the sampler leaves it).
T0_WINDOW: dict[str, tuple[float, float]] = {
    "shock_cooling": (-2.0, 1.0),
    "arnett": (-2.0, 1.0),
    "shock_cooling_and_arnett": (-2.0, 1.0),
    # peak_time, not an explosion time: the epoch of the template's phase zero,
    # which for these templates is maximum light. It has to reach wherever a
    # peak can sit, so this is far wider than an explosion window, and it is a
    # fixed offset like all the others rather than anything read off the data.
    # The same window NMMA puts on timeshift in its generic sncosmo prior.
    # Override with --t0-min / --t0-max, which bound peak_time on this path
    # since the model never reads t0.
    #
    # Note that ln Z is NOT a guide for choosing this: a narrower window scores
    # better purely because it encloses less prior volume. Measured on
    # SN2021ugl, (-5, +20) gives 1.80 and (-10, +120) gives 0.17, and the gap
    # of 1.63 is ln(130/25) = 1.65, i.e. the log of the width ratio and nothing
    # else. The posterior does not move. Pick the window from what a peak epoch
    # can physically be, not from the evidence it produces.
    SNCOSMO_MODEL: (-30.0, 30.0),
    # Slow transients, where the rise alone lasts weeks and the explosion can
    # genuinely predate the discovery by that much. Left wide on purpose.
    "slsn": (-150.0, 0.0),
    "tde_analytical": (-150.0, 0.0),
    "cooling_envelope": (-150.0, 0.0),
}

# The parameters the sncosmo path samples. redback ships no prior file for
# sncosmo_models itself (it warns "you may need to set up your prior
# explicitly") because the model is a bridge to the whole sncosmo library and
# each template has its own source parameters. It does ship one per *named*
# template wrapper, and those are the pattern this follows:
#
#   priors/salt2.prior              redshift, peak_time, t0, x0, x1, c
#   priors/sn1998bw_template.prior  redshift, amplitude
#
# so: redshift, a time origin, a peak epoch, and the template's own shape.
#
# ``peak_time_mjd`` rather than ``peak_time``: t0_base_model has a dedicated
# branch for it,
#
#     if 'peak_time_mjd' in kwargs:
#         kwargs['peak_time'] = kwargs['peak_time_mjd'] - t0
#
# which is redback's own way of sampling the peak as an absolute MJD alongside
# t0 and letting the wrapper reduce it to days since the explosion. Measured:
# phase zero of v19-1993j-corr is maximum light (ztfg maximum at phase -0.2 d),
# so peak_time is a peak, not an explosion.
# Keys common to every template. The template's own source parameters are added
# to these, read from sncosmo, because they differ: x0/x1/c for SALT, amplitude
# for the v19-* family.
#
# ``t0`` is deliberately NOT among them. redback's example samples it, but
# ``sncosmo_models(time, redshift, model_kwargs=None, **kwargs)`` never reads
# it, and that is measured, not assumed: two calls differing only in t0 return
# identical magnitudes to every decimal, while changing peak_time changes them.
# A sampled t0 would therefore be a flat direction, costing prior volume and
# buying nothing.
# redback's own default for rv_mw in its extinction models. A_V = R_V x E(B-V).
R_V_MW = 3.1

SNCOSMO_COMMON_KEYS: tuple[str, ...] = ("redshift", "peak_time", "ebv", "av_mw")

# sncosmo_models is called directly: it sets its own time origin with
# ``model.set(t0=peak_time)`` and adds its own extinction inside sncosmo, and
# redback's own example says so ("we can directly sample t0 without using a
# phase model"). t0_base_model would impose a second origin on the same array.
_NO_PHASE_MODEL = frozenset({SNCOSMO_MODEL})

# shock_cooling and shock_cooling_and_arnett are the models redback ships no
# BUILTIN_MODEL_METADATA for, so their family cannot be read off `model_type`.
_MODEL_TYPE_FALLBACK = {
    "shock_cooling": "shock_powered",
    "shock_cooling_and_arnett": "supernova",
}

# cooling_envelope's 'spectra' output reports time in seconds where every other
# model reports days, and the t0 extinction wrappers route magnitude output
# through 'spectra'. Measured against t0_base_model at zero extinction, post-t0:
# the four other models agree to 0.000000 mag, cooling_envelope diverges by 1.55.
_NO_EXTINCTION_WRAPPER = frozenset({"cooling_envelope"})


# redback's likelihoods for magnitude data, with what each one can and cannot
# do. None offers both a systematic budget and a survival term for upper
# limits, which NMMA does in a single function, so the caller has to choose.
#
#   sigma_i is the photometric error from the file; sigma is a sampled
#   parameter added to it by the two budget variants.
LIKELIHOODS = {
    "upper-limits": dict(
        cls="GaussianLikelihoodWithUpperLimits",
        sigma="sigma_i only",
        limits=True,
        note="redback's own choice when non-detections are present. Keeps "
        "them through a survival term, carries no systematic budget.",
    ),
    "quadrature": dict(
        cls="GaussianLikelihoodQuadratureNoise",
        sigma="sqrt(sigma_i**2 + sigma**2)",
        limits=False,
        note="an additive budget in magnitudes, the same formula NMMA applies "
        "behind --em-error-budget, with sigma sampled rather than fixed. "
        "Drops upper limits: no survival term.",
    ),
    "systematic": dict(
        cls="GaussianLikelihoodWithSystematicNoise",
        sigma="sqrt(sigma_i**2 + model**2 * sigma**2)",
        limits=False,
        note="a budget proportional to the model value, which is a fractional "
        "error on a FLUX. In magnitude mode model**2 is ~361 at mag 19, so "
        "the parameter is amplified and scales with brightness, which a "
        "logarithmic quantity should not do. Drops upper limits.",
    ),
}

DEFAULT_LIKELIHOOD = "upper-limits"


def likelihood_class(name: str):
    """The redback likelihood class named by a :data:`LIKELIHOODS` key."""
    import redback.likelihoods as L

    return getattr(L, LIKELIHOODS[name]["cls"])


def model_type(name: str) -> str | None:
    """redback's own family for ``name``, from BUILTIN_MODEL_METADATA."""
    from redback.model_metadata import BUILTIN_MODEL_METADATA

    entry = BUILTIN_MODEL_METADATA.get(name)
    found = getattr(entry, "model_type", None) if entry is not None else None
    return found or _MODEL_TYPE_FALLBACK.get(name)


def extinction_model(name: str) -> str | None:
    """redback's combined t0 + extinction model for ``name``, or None.

    Derived from redback's ``model_type`` rather than tabulated, so a family
    renamed or added upstream is picked up without an edit here.
    """
    from redback.model_library import all_models_dict

    if name in _NO_EXTINCTION_WRAPPER:
        return None
    family = model_type(name)
    if family is None:
        return None
    wrapper = f"t0_{family}_extinction"
    return wrapper if wrapper in all_models_dict else None


def supports_extinction(name: str) -> bool:
    """Whether ``name`` can carry sampled extinction through its t0 wrapper."""
    return extinction_model(name) is not None


def time_parameter(name: str) -> str:
    """The key carrying the model's time origin, in MJD.

    ``t0`` for redback's forward models, which take it through
    ``t0_base_model``. ``peak_time`` for the sncosmo wrapper, which is called
    directly and sets the template's phase zero with it instead; for the
    templates here that phase zero is maximum light, measured (ztfg maximum at
    phase -0.2 d for v19-1993j-corr).
    """
    return "peak_time" if name == SNCOSMO_MODEL else "t0"


def t0_window(
    name: str,
    reference_mjd: float | None = None,
    first_detection_mjd: float | None = None,
) -> tuple[float, float]:
    """The t0 prior window in days from the first epoch, derived from the data.

    :data:`T0_WINDOW` is a fixed table, and a fixed table cannot be right for
    every target: the gap between the first epoch of a file and its first
    detection is a property of the observing cadence, not of the model. It is
    3.99 d on SN2021ugl and 6.04 d on SN2020oi, so the same ``(-2, +1)`` lands
    on a different part of the light curve each time, and on SN2020oi it places
    t0 entirely before the deep pre-explosion limit that rules it out.

    Both bounds come from the data instead, with no per-target tuning:

    - the upper bound is the **first detection**. A transient has exploded
      before it is seen, and ``t0_base_model`` returns 1000 mag before ``t0``,
      so anything later is dead prior volume: it costs ln Z an Occam factor
      with no support from the data.
    - the lower bound is the **first epoch** of the file. There is no
      information before the data start. A posterior piling up against this
      edge is the signal that the explosion may predate the file, and it is
      visible in the figure rather than hidden.

    Models in :data:`_NO_PHASE_MODEL` keep the fixed table: their time
    parameter is ``peak_time``, maximum light rather than explosion, which the
    first detection does not bound at all and which may fall on either side of
    it.

    Falls back to the table when the caller passes no epochs, and when the
    first epoch *is* the first detection: the file then carries nothing before
    the transient, so the model's own slack is all there is to go on, capped at
    the detection.

    Parameters
    ----------
    name : str
        Base model key.
    reference_mjd, first_detection_mjd : float, optional
        From :func:`_data.first_epoch_mjd` and
        :func:`_data.first_detection_mjd`.

    Returns
    -------
    tuple of float
        ``(t0_min, t0_max)``, days relative to ``reference_mjd``.
    """
    fallback = T0_WINDOW[name]
    if name in _NO_PHASE_MODEL or reference_mjd is None or first_detection_mjd is None:
        return fallback

    span = float(first_detection_mjd) - float(reference_mjd)
    if span <= 0.0:
        return (fallback[0], 0.0)
    return (0.0, span)


def sncosmo_source_parameters(template: str) -> tuple[str, ...]:
    """The template's own parameters, read from sncosmo rather than tabulated.

    ``salt2`` and ``salt3`` expose ``x0, x1, c``; the ``v19-*`` templates
    expose ``amplitude`` alone. redback passes them through
    ``model_kwarg_names``, which is how its own example samples them, so
    whatever ``--sncosmo-model`` accepts is covered without an edit here.

    Parameters
    ----------
    template : str
        An sncosmo source name.

    Returns
    -------
    tuple of str
    """
    import sncosmo

    return tuple(sncosmo.Model(source=template).source.param_names)


def sncosmo_max_redshift(template: str, bands) -> float:
    """Largest redshift at which ``template`` still covers every band.

    An sncosmo source is defined over a rest-frame wavelength range, and at
    redshift z it spans ``[minwave * (1+z), maxwave * (1+z)]`` in the observer
    frame. Outside that, ``bandmag`` does not return NaN, it **raises**::

        ValueError: bandpass 'ztfg' [3676, .., 5608] outside spectral range
                    [3719.1, .., 25489.2]

    so a redshift prior wider than this kills the fit outright rather than
    scoring badly. The binding edge is usually the red one: v19-1993j-corr
    covers 1605-11000 A at rest, and ATLAS o reaches far enough into the red
    that z = 0.33 is the limit.

    Parameters
    ----------
    template : str
        An sncosmo source name.
    bands : iterable of str
        The bands actually being fitted, as sncosmo knows them.

    Returns
    -------
    float
        ``inf`` when no band is known to sncosmo, leaving the prior untouched.
    """
    import sncosmo

    source = sncosmo.Model(source=template).source
    limits = []
    for band in {str(b) for b in bands}:
        try:
            bandpass = sncosmo.get_bandpass(band)
        except Exception:
            continue
        limits.append(bandpass.minwave() / source.minwave() - 1.0)
        limits.append(source.maxwave() / bandpass.maxwave() - 1.0)
    # float(), not the numpy scalar min() returns: bilby writes a prior bound
    # with repr(), so a numpy scalar lands in the effective prior file as
    # "np.float64(0.33...)", and reading that file back raises
    # ModuleNotFoundError: No module named 'np'. The effective prior is meant
    # to be replayable through --priors, so it has to stay plain Python.
    return float(min(limits)) if limits else float("inf")


def sncosmo_prior_stem(template: str) -> str:
    """Basename of the versioned prior file for ``template``.

    Two files, not one per template and not one for all: ``x0, x1, c`` and
    ``amplitude`` are mutually exclusive key sets, so a single file carrying
    the SALT keys makes a v19 fit fail outright, while splitting per template
    would be five near-identical files to keep in step.
    """
    return (
        "sncosmo_salt"
        if "x0" in sncosmo_source_parameters(template)
        else "sncosmo_generic"
    )


def sncosmo_keys(template: str) -> tuple[str, ...]:
    """Every key the sncosmo path samples for ``template``.

    The template's own source parameters are sampled as they are, which is the
    only brightness route redback actually exercises: ``model_kwarg_names``
    with ``x0, x1, c`` in its example and in ``priors/salt2.prior``, and with
    ``amplitude`` in ``priors/sn1998bw_template.prior``.

    ``sncosmo_models`` also accepts ``use_set_peak_magnitude``, which would
    anchor the brightness on a peak absolute magnitude instead and so avoid
    priming a raw flux normalisation. It is deliberately NOT used: nothing in
    redback ever sets it to True, so it is a provision, not a practice.
    """
    return SNCOSMO_COMMON_KEYS + sncosmo_source_parameters(template)


def sampled(name: str) -> tuple[str, ...]:
    """The model's sampled parameter keys, in prior order, from redback.

    ``sncosmo_models`` is the exception: redback ships no prior for it, so only
    its template-independent keys are known here
    (:data:`SNCOSMO_COMMON_KEYS`). For the full list, which depends on the
    chosen template, call :func:`sncosmo_keys`.
    """
    from redback.priors import get_priors

    if name == SNCOSMO_MODEL:
        return SNCOSMO_COMMON_KEYS
    return tuple(get_priors(name).keys())


def constraint_keys(name: str) -> frozenset[str]:
    """Names of the ``Constraint`` priors redback attaches to ``name``.

    These are what bilby's ``PriorDict.evaluate_constraints`` actually tests. A
    conversion function alone rejects nothing: without a matching ``Constraint``
    prior the computed ratio is discarded.
    """
    import bilby
    from redback.priors import get_priors

    priors = get_priors(name, constraint=True)
    return frozenset(
        key
        for key, prior in priors.items()
        if isinstance(prior, bilby.core.prior.Constraint)
    )


def has_constraints(name: str) -> bool:
    """Whether redback ships an enforceable physical constraint for ``name``."""
    return bool(constraint_keys(name))


def observed_model(name: str):
    """redback's t0 model for ``name``, with extinction where one works.

    A router, nothing more: ``t0_<family>_extinction`` when redback has a usable
    one, bare ``t0_base_model`` otherwise. The model's output is returned as
    redback computes it.

    Parameters
    ----------
    name : str
        Base model key, as redback knows it.

    Returns
    -------
    callable
        ``model(time_mjd, t0=..., **kwargs)``. Pass ``base_model=name`` and
        ``output_format="magnitude"`` in ``model_kwargs``; ``t0_base_model``
        reads both from kwargs.

    Notes
    -----
    ``t0_base_model`` returns ``concatenate((pre_t0, post_t0))``, so the time
    array must be sorted ascending or the output is misaligned with the data.

    Models in :data:`_NO_PHASE_MODEL` are returned as they are: they carry
    their own time origin, so a phase wrapper would impose a second one.
    """
    from redback.model_library import all_models_dict

    if name in _NO_PHASE_MODEL:
        return _with_galactic_extinction(
            _reject_outside_template(all_models_dict[name])
        )
    return all_models_dict[extinction_model(name) or "t0_base_model"]


# Effective wavelength per band, in Angstrom. Filled on demand from sncosmo's
# own bandpass registry, the same one the template is evaluated against.
_WAVE_EFF: dict[str, float] = {}


def _wave_eff(band: str) -> float:
    """Effective wavelength of a band, cached, from sncosmo's registry."""
    if band not in _WAVE_EFF:
        import sncosmo

        _WAVE_EFF[band] = float(sncosmo.get_bandpass(band).wave_eff)
    return _WAVE_EFF[band]


def _with_galactic_extinction(function):
    """Redden the model by a sampled Galactic ``av_mw``, as NMMA does.

    redback's ``sncosmo_models`` registers ``sncosmo.F99Dust()`` as the 'mw'
    effect but never sets ``mwebv``, and the string appears nowhere in redback,
    so the Galactic term is inert on that path. It cannot be supplied either:
    ``model.update(_model_kwargs)`` runs *before* ``add_effect``, so the
    parameter does not exist yet when redback reads its kwargs.

    Without it the reddening does not vanish, it is reassigned: measured on
    SN2021ugl, the host ``ebv`` drops by 0.069 when the Galactic 0.0637 is
    removed from the data, so the host parameter was absorbing it almost one
    for one and was being reported as host extinction.

    This closes that gap the same way NMMA does in
    ``EMModel.apply_extinction_correction``: a per-filter ``A_lambda`` added to
    the model magnitudes, with ``E(B-V)`` sampled under a dust-map bound rather
    than fixed. Two deliberate differences from NMMA, both for consistency
    inside this pipeline rather than with it:

    - the curve is ``fitzpatrick99``, redback's own default ``mw_law``, not
      NMMA's G23. Measured over the bands used here the two differ by at most
      0.0084 mag, against photometric errors of 0.113, so the choice is
      numerically irrelevant and internal consistency decides it.
    - the parameter is ``av_mw`` in magnitudes of A_V, redback's own
      parameterisation for the physical models, not NMMA's ``Ebv``. They are
      the same quantity through A_V = R_V x E(B-V).

    ``av_mw`` is consumed here and never reaches ``sncosmo_models``, which
    would pass an unknown name to ``model.update()``.
    """
    import functools

    @functools.wraps(function)
    def wrapped(time, *args, **kwargs):
        av_mw = float(kwargs.pop("av_mw", 0.0) or 0.0)
        result = function(time, *args, **kwargs)
        if av_mw == 0.0:
            return result

        import extinction

        bands = np.atleast_1d(kwargs.get("bands"))
        waves = np.array([_wave_eff(str(b)) for b in bands], dtype=float)
        a_lambda = extinction.fitzpatrick99(waves, av_mw, R_V_MW)
        if kwargs.get("output_format") == "magnitude":
            return result + a_lambda
        return result * 10.0 ** (-0.4 * a_lambda)

    return wrapped


def _reject_outside_template(function):
    """Turn an out-of-range sncosmo call into a rejected sample, not a crash.

    An sncosmo source covers a fixed rest-frame wavelength range, and past the
    redshift where a bandpass leaves it, ``bandmag`` raises rather than
    returning NaN::

        ValueError: bandpass 'ztfg' [3676, .., 5608] outside spectral range

    Truncating the redshift prior to avoid that would be the wrong fix: it
    removes prior volume, which inflates ln Z by the log of the width removed
    (here ln(3 / 0.33) = 2.2 nats) and so flatters the model for a reason that
    has nothing to do with the data. Returning an infinite magnitude instead
    leaves the prior as declared and lets the likelihood reject the sample,
    which is what an unevaluable model deserves.
    """
    import functools

    @functools.wraps(function)
    def wrapped(time, *args, **kwargs):
        try:
            return function(time, *args, **kwargs)
        except ValueError as exc:
            if "outside spectral range" not in str(exc):
                raise
            return np.full(np.shape(np.atleast_1d(time)), np.inf)

    return wrapped


def model_kwargs(name: str, bands, *, sncosmo_model: str | None = None) -> dict:
    """The non-sampled keyword arguments redback's model needs.

    Shared by the fit and the plot so the curve drawn is the curve fitted.

    Parameters
    ----------
    name : str
        Base model key.
    bands : sequence of str
        One band per epoch, aligned with the data. ``t0_base_model`` masks
        these per epoch with ``_mask_per_epoch_kwargs``, so they must repeat
        with the data rather than be a unique set.
    sncosmo_model : str, optional
        sncosmo template name, required for ``sncosmo_models`` and ignored
        otherwise. redback defaults it to ``salt2``.

    Returns
    -------
    dict
        Ready to pass as ``model_kwargs`` to ``redback.fit_model``.

    Notes
    -----
    On the sncosmo path ``use_set_peak_magnitude=True`` makes the brightness a
    peak absolute magnitude instead of the template's raw ``amplitude``, which
    for a Type IIb at z = 0.04 sits around 1e-13 and is awkward to prior.

    ``model_kwarg_names=[]`` is the empty case of the wrapper's own mechanism
    for sampling an sncosmo source parameter: with ``model_kwargs`` left at
    None it reads ``kwargs['model_kwarg_names']`` and pulls each name from
    kwargs, so an empty list means "take nothing from the sampler" and
    ``model.update({})`` is a no-op. Passing ``model_kwargs={}`` here instead
    would collide: redback forwards this dict to the model as ``**kwargs``, and
    ``model_kwargs`` is also the wrapper's own third parameter, so the model
    would get two values for it.

    redback adds its Milky Way extinction effect after ``model.update()``, and
    nothing then sets ``mwebv``, so sncosmo's default of 0.0 stands and the
    effect is inert, so ``av_mw`` is applied by
    :func:`_with_galactic_extinction` around the model instead of by redback,
    and is deliberately absent from ``model_kwarg_names``: ``model.update()``
    would reject a name the template does not define.
    """
    from redback.utils import bands_to_frequency

    kwargs = {
        "output_format": "magnitude",
        "bands": bands,
        "frequency": bands_to_frequency(list(bands)),
    }
    if name == SNCOSMO_MODEL:
        sampled_source = sncosmo_source_parameters(sncosmo_model)
        kwargs.update(
            sncosmo_model=sncosmo_model,
            model_kwarg_names=list(sampled_source),
        )
    else:
        kwargs["base_model"] = name
    return kwargs


def describe(name: str) -> str:
    """Summary of a model, from redback's own metadata and citation."""
    from redback.model_library import all_models_dict

    citation = getattr(all_models_dict[name], "citation", None)
    return (
        f"{name} (model_type={model_type(name)})\n"
        f"Sampled: {', '.join(sampled(name))}\n"
        f"Extinction: {extinction_model(name) or 'none, no usable wrapper'}\n"
        f"Constraints: {', '.join(sorted(constraint_keys(name))) or 'none'}\n"
        f"Reference: {citation or 'not recorded by redback'}"
    )


def check_against_redback(prior_dir=None) -> dict[str, list[str]]:
    """Report any model this project names that redback no longer provides.

    Run after a redback upgrade. Nothing about the models is stored here, so the
    only drift possible is a model, a t0 wrapper or a prior key disappearing.

    Parameters
    ----------
    prior_dir : pathlib.Path, optional
        Directory of ``<model>.prior`` files to validate against redback's own
        keys. None skips that check.

    Returns
    -------
    dict
        Model name (or ``"redback"``) -> list of problems. Empty when in sync.
    """
    from redback.model_library import all_models_dict

    problems: dict[str, list[str]] = {}

    if "t0_base_model" not in all_models_dict:
        problems["redback"] = ["t0_base_model is gone from all_models_dict"]

    for name in MODELS:
        messages: list[str] = []
        if name not in all_models_dict:
            problems[name] = ["model absent from redback's all_models_dict"]
            continue
        try:
            keys = sampled(name)
        except Exception as exc:
            messages.append(f"prior unavailable: {exc}")
            keys = ()
        if keys and prior_dir is not None:
            prior_file = prior_dir / f"{name}.prior"
            if prior_file.exists():
                import bilby

                unknown = set(bilby.core.prior.PriorDict(str(prior_file))) - set(keys)
                if unknown:
                    messages.append(
                        f"{prior_file.name} names keys redback does not have: "
                        f"{sorted(unknown)}"
                    )
        if messages:
            problems[name] = messages

    return problems
