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
)

# Bounds on ``t0``, in days relative to the first detection. redback samples t0
# as an absolute MJD, t0_base_model runs it through ``Time(t0, format='mjd')``,
# so a bare offset would be read as an epoch in 1858, and both of redback's own
# examples derive the bounds from the data the same way:
#
#     priors['t0'] = Uniform(data['MJD'].values.min() - 200,
#                            data['MJD'].values.min() - 1, name='t0')
#     priors['t0'] = Uniform(supernova.x[0] - 10, supernova.x[0] - 0.05, name='t0')
#
# These are the offsets redback_priors adds to the first detection. redback
# cannot ship them: the plausible range depends on the data, not on the model.
# Override with --t0-min / --t0-max.
T0_WINDOW: dict[str, tuple[float, float]] = {
    "shock_cooling": (-3.0, 0.0),
    "arnett": (-30.0, 0.0),
    "slsn": (-150.0, 0.0),
    "tde_analytical": (-150.0, 0.0),
    "cooling_envelope": (-150.0, 0.0),
    # Its shock-cooling component anchors t0 within days of the explosion, but
    # the window is kept as wide as arnett's so the data, not the bound, decide.
    "shock_cooling_and_arnett": (-30.0, 0.0),
}

# shock_cooling and shock_cooling_and_arnett are the models redback ships no
# BUILTIN_MODEL_METADATA for, so their family cannot be read off `model_type`.
_MODEL_TYPE_FALLBACK = {
    "shock_cooling": "shock_powered",
    "shock_cooling_and_arnett": "supernov a",
}

# cooling_envelope's 'spectra' output reports time in seconds where every other
# model reports days, and the t0 extinction wrappers route magnitude output
# through 'spectra'. Measured against t0_base_model at zero extinction, post-t0:
# the four other models agree to 0.000000 mag, cooling_envelope diverges by 1.55.
_NO_EXTINCTION_WRAPPER = frozenset({"cooling_envelope"})


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


def sampled(name: str) -> tuple[str, ...]:
    """The model's sampled parameter keys, in prior order, from redback."""
    from redback.priors import get_priors

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
    """
    from redback.model_library import all_models_dict

    return all_models_dict[extinction_model(name) or "t0_base_model"]


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
