"""Guards for the redback pipeline.

Each test here pins down a failure mode that is silent rather than loud: a
constraint that computes and is then ignored, a NaN that scores the best
possible likelihood, an unsorted time array that misaligns the model from the
data. None of those raise on their own, so only a test catches them.

Run the fast ones with::

    .venv-redback/bin/python -m pytest tests -m "not slow"

The slow marker covers the injection-recovery test, which samples.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from carich._paths import DATA, PRIORS_REDBACK, root

sys.path.insert(0, str(root() / "src" / "fitting" / "redback-fit"))

import redback_data as rd  # noqa: E402
import redback_models as rm  # noqa: E402
import redback_priors  # noqa: E402

CONSTRAINED_MODELS = ("slsn", "tde_analytical", "cooling_envelope")
DATAFILE = DATA / "processed" / "SN2021ugl.dat"
PRIOR_DIR = PRIORS_REDBACK
REFERENCE_MJD = rd.first_epoch_mjd(rd.read_photometry(DATAFILE))


@pytest.fixture(autouse=True)
def _deterministic_prior_draws():
    """Reseed bilby's RNG before every test.

    ``np.random.seed`` does **not** control ``PriorDict.sample``: bilby draws
    from its own generator (``bilby.core.utils.random``). Seeding only numpy
    leaves every test that samples a prior at the mercy of whatever RNG state
    the tests before it happened to leave, so a suite that passes in one order
    fails in another, which is how this fixture came to exist.

    ``fit_redback.main`` seeds both for the same reason.
    """
    import bilby

    bilby.core.utils.random.seed(0)
    np.random.seed(0)


@pytest.fixture(scope="module")
def rows():
    return rd.read_photometry(DATAFILE)


def _config(**overrides):
    """A RunConfig with every field set, for the prior-building tests."""
    import fit_redback

    defaults = dict(
        object_name="SN2021ugl",
        datafile=DATAFILE,
        outdir=Path("/tmp/redback-tests"),
        model="slsn",
        priors=None,
        kappa=0.07,
        z_known=0.0412,
        z_phot=None,
        z_phot_err=None,
        ebv_mw=0.0637,
        ebv_source="user",
        sample_av_host=False,
        av_host_max=1.0,
        t0_min=-150.0,
        t0_max=0.0,
        sncosmo_model="v19-1993j-corr",
        likelihood="upper-limits",
        sigma_sys_max=0.5,
        detections_only=False,
        plot_theme="print",
        data_tmin=None,
        data_tmax=None,
        upper_limit_sigma=3.0,
        nlive=100,
        sampler="dynesty",
        no_constraint=False,
        seed=42,
    )
    defaults.update(overrides)
    return fit_redback.RunConfig(**defaults)


# --------------------------------------------------------------------------
# The t0 wrapper
# --------------------------------------------------------------------------


# sncosmo_models reaches t0_base_model like the rest, but redback ships it no
# prior and no metadata, so the tests that read get_priors(model) or assert a
# shipped prior file are parametrised over the others.
FORWARD_MODELS = tuple(m for m in rm.MODELS if m != rm.SNCOSMO_MODEL)


@pytest.mark.parametrize("model", list(FORWARD_MODELS))
def test_t0_wrapper_pre_explosion(model):
    """1000 mag before t0, finite after, for every catalogued model."""
    from redback.priors import get_priors
    from redback.utils import bands_to_frequency

    np.random.seed(0)
    parameters = get_priors(model).sample()
    parameters["redshift"] = 0.05

    t0 = 59400.0
    times = np.array([t0 - 5.0, t0 - 1.0, t0 + 2.0, t0 + 10.0])
    bands = ["ztfg"] * times.size

    extras = {"av_host": 0.0, "av_mw": 0.0} if rm.supports_extinction(model) else {}
    function = rm.observed_model(model)
    out = np.asarray(
        function(
            times,
            t0=t0,
            base_model=model,
            output_format="magnitude",
            bands=bands,
            frequency=bands_to_frequency(bands),
            **extras,
            **parameters,
        )
    )

    sentinel = 5000.0 if rm.supports_extinction(model) else 1000.0

    assert out.shape == times.shape
    assert np.all(out[:2] == sentinel), "pre-t0 epochs must be the sentinel"
    assert np.all(np.isfinite(out[2:])), "post-t0 epochs must be finite"


# --------------------------------------------------------------------------
# Data handling
# --------------------------------------------------------------------------


def test_unsorted_times_rejected(rows):
    """t0_base_model concatenates pre/post t0, so unsorted input must raise."""
    swapped = [rows[5], rows[0]] + list(rows[1:5]) + list(rows[6:])
    with pytest.raises(ValueError, match="not sorted"):
        rd.build_transient(swapped, "X", upper_limit_sigma=3.0)


def test_no_detection_rejected(rows):
    limits_only = [{**row, "detected": False} for row in rows]
    with pytest.raises(ValueError, match="no detection"):
        rd.build_transient(limits_only, "X", upper_limit_sigma=3.0)


def test_measured_errors_pass_through_untouched(rows):
    """No systematic floor is folded into the data, as in redback's own loader.

    An earlier version added NMMA's sigma_eb in quadrature, which compressed the
    relative weighting of this data set by a factor of 14: the best detection
    (0.043 mag) outweighs the worst (0.207) by 23x on the measured errors, but
    only 1.6x once a 0.25 mag floor dominates both.
    """
    transient = rd.build_transient(rows, "SN2021ugl", upper_limit_sigma=3.0)
    detections = np.asarray(transient.detections, dtype=bool)
    errors = np.asarray(transient.y_err, dtype=float)

    reported = np.array(
        [row["mag_err"] for row in rows if row["detected"]], dtype=float
    )
    assert np.allclose(np.sort(errors[detections]), np.sort(reported)), (
        "detection errors must be exactly what the file reported"
    )
    # Limits keep their NaN: the survival term never reads it, and the figures
    # draw them as arrows without an error bar.
    assert np.all(np.isnan(errors[~detections]))


def test_redback_keeps_the_upper_limit_likelihood(rows):
    """What the guard above protects: the limits survive into the likelihood."""
    transient = rd.build_transient(rows, "SN2021ugl", upper_limit_sigma=3.0)
    _, _, y, _, detections = transient.get_filtered_data_with_limits()

    assert detections is not None
    assert np.all(np.isfinite(y[~detections])), (
        "a non-finite limit magnitude makes redback fall back to a "
        "detections-only GaussianLikelihood"
    )
    assert int(np.sum(~detections)) > 0


def test_upper_limit_sigma_is_the_sigma_level(rows):
    """upper_limit_sigma carries redback's meaning: the sigma level of a limit.

    It used to be inverted into (2.5/ln10)/sigma_eb so that the survival width
    would equal NMMA's error budget, a repurposing of the parameter, made to
    keep the two frameworks' evidences on one scale. They are not compared, so
    the parameter now means what redback says it means, and the value is a
    property of the data: the ingest step writes ATLAS 5-sigma limits.
    """
    transient = rd.build_transient(rows, "SN2021ugl", upper_limit_sigma=3.0)
    assert transient.upper_limit_sigma == 3.0
    assert transient.has_upper_limits
    # The survival width redback derives from it.
    survival_width = (2.5 / np.log(10.0)) / transient.upper_limit_sigma
    assert survival_width == pytest.approx(0.3619, abs=1e-4)


def test_transient_is_on_mjd_scale(rows):
    """The likelihood reads `time_mjd`, so Transient.x must be MJD."""
    transient = rd.build_transient(rows, "SN2021ugl", upper_limit_sigma=3.0)
    assert transient.x.min() > 50000.0, "x should be MJD, not a phase"
    assert np.all(np.diff(transient.x) >= 0)
    assert transient.first_detection_mjd == pytest.approx(rd.first_detection_mjd(rows))


def test_data_window_relative_to_the_first_epoch(rows):
    """The window is anchored where redback anchors it: the first epoch.

    ``supernova.x[0]`` in redback's examples, upper limits included, not the
    first detection. On SN2021ugl the two are 3.99 days apart, which with a
    three-day t0 window is the difference between two disjoint priors.
    """
    epoch = rd.first_epoch_mjd(rows)
    detection = rd.first_detection_mjd(rows)
    assert epoch < detection, "this file has pre-detection limits, as expected"

    kept = rd.build_transient(rows, "X", upper_limit_sigma=3.0, data_tmin=0.0)
    assert kept.x.min() >= epoch
    # Anchored on the epoch, data_tmin=0 keeps the pre-detection limits.
    assert kept.x.min() < detection

    trimmed = rd.build_transient(rows, "X", upper_limit_sigma=3.0, data_tmin=10.0)
    assert trimmed.x.min() >= epoch + 10.0
    assert len(trimmed.x) < len(kept.x)


def test_first_detection_is_display_only(rows):
    """It is recorded and plotted, never used to anchor the fit."""
    import inspect

    import fit_redback

    transient = rd.build_transient(rows, "X", upper_limit_sigma=3.0)
    assert transient.first_detection_mjd == pytest.approx(rd.first_detection_mjd(rows))
    assert transient.first_epoch_mjd == pytest.approx(rd.first_epoch_mjd(rows))

    # The runner must anchor on the epoch, not the detection.
    source = inspect.getsource(fit_redback.main)
    assert "first_epoch_mjd(rows)" in source
    assert "first_detection_mjd(rows)" not in source


# --------------------------------------------------------------------------
# Priors and constraints, the bug this pipeline used to have
# --------------------------------------------------------------------------


@pytest.mark.parametrize("model", CONSTRAINED_MODELS)
def test_constraints_attached(model, rows):
    """Constraint priors must be present, or bilby rejects nothing."""
    import bilby
    import redback_priors

    cfg = _config(model=model, t0_min=rm.T0_WINDOW[model][0])
    priors = redback_priors.build_priors(cfg, REFERENCE_MJD)

    constraints = {
        key
        for key, prior in priors.items()
        if isinstance(prior, bilby.core.prior.Constraint)
    }
    assert constraints == set(rm.constraint_keys(model))


def test_slsn_rejects_unphysical_sample(rows):
    """A slow magnetar with a huge ejecta mass must be rejected.

    Also exercises _with_fixed_parameters: slsn_constraint reads ``kappa``,
    which is a DeltaFunction here and so absent from the sampled dict.
    """
    import bilby
    import redback_priors

    cfg = _config(model="slsn", kappa=0.07)
    priors = redback_priors.build_priors(cfg, REFERENCE_MJD)
    constrained = {
        key
        for key, prior in priors.items()
        if isinstance(prior, bilby.core.prior.Constraint)
    }

    unphysical = {
        "p0": 10.0,
        "mej": 100.0,
        "vej": 30000.0,
        "bp": 1.0,
        "mass_ns": 1.4,
        "theta_pb": 0.1,
        "kappa_gamma": 0.1,
        "temperature_floor": 5000.0,
        "t0": priors["t0"].minimum,
    }
    assert priors.evaluate_constraints(unphysical) == 0

    drawn = {k: v for k, v in priors.sample().items() if k not in constrained}
    assert priors.evaluate_constraints(drawn) == 1


@pytest.mark.parametrize("model", ("arnett",) + CONSTRAINED_MODELS)
def test_sampler_can_draw_constrained_samples(model, rows):
    """The path bilby actually takes before sampling, not a hand-built dict.

    ``Sampler._verify_parameters`` calls
    ``priors.sample_subset_constrained_as_array``, which calls
    ``evaluate_constraints`` with arrays, and bilby sizes its accumulator from
    the first entry of the conversion's output. A conversion that returns the
    pinned scalars first makes that accumulator 0-d and the whole fit dies with
    "non-broadcastable output operand", after the priors have been logged, so
    the traceback points at bilby and not at us.

    Every earlier constraint test built its sample by hand and never exercised
    this, and every smoke test used shock_cooling, the one model with no
    constraints at all. Hence this one.
    """
    import bilby

    cfg = _config(
        model=model,
        kappa=0.07,
        t0_min=rm.T0_WINDOW[model][0],
    )
    priors = redback_priors.build_priors(cfg, REFERENCE_MJD)
    keys = [
        key
        for key, prior in priors.items()
        if not isinstance(prior, bilby.core.prior.Constraint)
    ]

    drawn = priors.sample_subset_constrained(keys=keys, size=1)
    assert set(drawn) == set(keys)
    for key, value in drawn.items():
        assert np.shape(value) == (1,), f"{key} came back as {np.shape(value)}"


def test_no_constraint_flag(rows):
    import bilby
    import redback_priors

    cfg = _config(model="slsn", no_constraint=True)
    priors = redback_priors.build_priors(cfg, REFERENCE_MJD)
    assert not [
        key
        for key, prior in priors.items()
        if isinstance(prior, bilby.core.prior.Constraint)
    ]


def test_prior_file_override_keeps_constraints(rows, tmp_path):
    """Overriding one key must not drop the conversion function."""
    import bilby
    import redback_priors

    prior_file = tmp_path / "slsn.prior"
    prior_file.write_text("p0 = Uniform(minimum=1.0, maximum=2.0, name='p0')\n")

    cfg = _config(model="slsn", priors=prior_file)
    priors = redback_priors.build_priors(cfg, REFERENCE_MJD)

    assert priors["p0"].minimum == 1.0
    assert priors["p0"].maximum == 2.0
    assert [
        key
        for key, prior in priors.items()
        if isinstance(prior, bilby.core.prior.Constraint)
    ]


def test_unknown_prior_key(rows, tmp_path):
    import redback_priors

    prior_file = tmp_path / "bad.prior"
    prior_file.write_text("not_a_parameter = Uniform(minimum=0, maximum=1, name='x')\n")

    cfg = _config(model="slsn", priors=prior_file)
    with pytest.raises(ValueError, match="does not have"):
        redback_priors.build_priors(cfg, REFERENCE_MJD)


def test_t0_prior_window(rows):

    reference = rd.first_epoch_mjd(rows)
    cfg = _config(model="arnett", t0_min=-30.0, t0_max=0.0)
    priors = redback_priors.build_priors(cfg, REFERENCE_MJD)

    # An absolute MJD, anchored on the first detection, redback's own form,
    # the same way its examples write Uniform(data['MJD'].min() - 200, ...).
    assert priors["t0"].name == "t0"
    assert priors["t0"].minimum == pytest.approx(reference - 30.0)
    assert priors["t0"].maximum == pytest.approx(reference)
    assert "timeshift" not in priors


def test_redshift_modes(rows):
    import bilby
    import redback_priors

    fixed = redback_priors.build_priors(_config(z_known=0.0412), REFERENCE_MJD)
    assert isinstance(fixed["redshift"], bilby.core.prior.DeltaFunction)

    photometric = redback_priors.build_priors(
        _config(z_known=None, z_phot=0.05, z_phot_err=0.01), REFERENCE_MJD
    )
    assert isinstance(photometric["redshift"], bilby.core.prior.TruncatedGaussian)

    sampled = redback_priors.build_priors(_config(z_known=None), REFERENCE_MJD)
    assert not isinstance(sampled["redshift"], bilby.core.prior.DeltaFunction)


def test_kappa_optional(rows):
    """--kappa pins; its absence leaves redback's prior alone."""
    import bilby
    import redback_priors

    pinned = redback_priors.build_priors(
        _config(model="arnett", kappa=0.07), REFERENCE_MJD
    )
    assert isinstance(pinned["kappa"], bilby.core.prior.DeltaFunction)
    assert pinned["kappa"].peak == 0.07

    free = redback_priors.build_priors(
        _config(model="arnett", kappa=None), REFERENCE_MJD
    )
    assert not isinstance(free["kappa"], bilby.core.prior.DeltaFunction)


# --------------------------------------------------------------------------
# Catalogue consistency
# --------------------------------------------------------------------------


def test_check_against_redback_clean():
    """No divergence between the catalogue and the installed redback."""
    assert rm.check_against_redback(prior_dir=PRIOR_DIR) == {}


def test_nothing_is_pinned_in_the_versioned_priors():
    """Every file is an exact copy; nothing here narrows a redback prior.

    nn and delta were pinned at the Piro, Haynie & Yao (2021) values for a
    while, on the argument that the paper fixes them. The argument held, the
    practice did not: a point prior where redback has Uniform(8, 12) and
    Uniform(1, 1.5) removes two dimensions, which lowers that model's Occam
    penalty and flatters its ln Z against any model fitted with its priors
    intact. Pass --priors with your own file if you want that; the versioned
    copies stay faithful.
    """
    import bilby

    for model in FORWARD_MODELS:
        priors = bilby.core.prior.PriorDict(str(PRIOR_DIR / f"{model}.prior"))
        pinned = [
            key
            for key, prior in priors.items()
            if isinstance(prior, bilby.core.prior.DeltaFunction)
        ]
        assert not pinned, f"{model}.prior pins {pinned}"


# Keys priors_redback overrides on top of redback's. Empty on purpose: every
# file is a faithful copy, and a narrowing belongs in a --priors file of your
# own, not in the versioned ones.
DELIBERATE_OVERRIDES: set = set()


@pytest.mark.parametrize("model", list(FORWARD_MODELS))
def test_prior_files_match_redback(model):
    """Each versioned prior file must reproduce redback's, bar the pinned keys.

    The files are regenerated copies, so the risk is silent drift: a redback
    upgrade moves a bound and the versioned copy keeps fitting the old one.
    This fails instead.
    """
    import os

    import bilby
    import redback
    from redback.priors import get_priors

    shipped = bilby.core.prior.PriorDict(
        os.path.join(os.path.dirname(redback.__file__), "priors", f"{model}.prior")
    )
    versioned = bilby.core.prior.PriorDict(str(PRIOR_DIR / f"{model}.prior"))

    assert set(versioned) == set(shipped), "prior keys differ from redback's"
    # get_priors is the path the code actually uses; the file on disk must agree.
    assert set(versioned) == set(get_priors(model))

    for key in shipped:
        if (model, key) in DELIBERATE_OVERRIDES:
            continue
        assert type(versioned[key]) is type(shipped[key]), key
        assert versioned[key].minimum == shipped[key].minimum, key
        assert versioned[key].maximum == shipped[key].maximum, key
        assert versioned[key].latex_label == shipped[key].latex_label, key


def test_prior_files_carry_provenance():
    """Every versioned prior file must say where it came from.

    These are copies of upstream data; an unlabelled copy is the thing a reader
    cannot tell from a hand-edited one.
    """
    for model in FORWARD_MODELS:
        text = (PRIOR_DIR / f"{model}.prior").read_text()
        assert text.lstrip().startswith("#"), f"{model}.prior has no header"
        assert "get_priors" in text, f"{model}.prior does not name its source"

    # Every file here is a copy of a redback prior. There is deliberately none
    # for sncosmo_models: see test_sncosmo_has_no_prior_file_of_its_own.
    assert not (PRIOR_DIR / f"{rm.SNCOSMO_MODEL}.prior").exists()


def test_every_model_has_a_description():
    """A model in MODELS with no description cannot be run at all.

    fit_redback logs ``describe(cfg.model)`` before sampling, so a missing
    entry is a KeyError two seconds into the run, not a cosmetic gap. Both
    shock_cooling_and_arnett and sncosmo_models were added to MODELS without
    one, and each aborted its first fit.
    """
    from model_descriptions import DESCRIPTIONS, describe

    for model in rm.MODELS:
        assert model in DESCRIPTIONS, f"{model} has no entry in DESCRIPTIONS"
        entry = DESCRIPTIONS[model]
        for field in ("label", "description", "derived", "reference", "caveat"):
            assert entry.get(field), f"{model}: {field} is missing or empty"
        # The call fit_redback actually makes, which also joins redback's
        # live parameter list, so an unknown key would raise here too. For the
        # sncosmo path that list is the template-independent part only.
        assert describe(model)


def test_model_type_fallback_names_a_real_redback_family():
    """Every hand-written family must resolve to a wrapper redback has.

    These two entries exist because redback ships no metadata for the models,
    so nothing upstream validates them. A typo here does not raise: the
    wrapper lookup simply misses, extinction_model returns None, and the model
    silently loses its extinction. That is exactly what a stray space in
    "supernova" once did to shock_cooling_and_arnett.
    """
    from redback.model_library import all_models_dict

    for model, family in rm._MODEL_TYPE_FALLBACK.items():
        assert family == family.strip(), f"{model}: {family!r} is not trimmed"
        assert f"t0_{family}_extinction" in all_models_dict, (
            f"{model}: redback has no t0_{family}_extinction"
        )


def test_every_model_resolves_its_extinction_or_says_why():
    """No model may lose extinction by accident, only by a recorded decision."""
    expected_without = set(rm._NO_EXTINCTION_WRAPPER) | {rm.SNCOSMO_MODEL}
    for model in rm.MODELS:
        if model in expected_without:
            assert not rm.supports_extinction(model), model
        else:
            assert rm.supports_extinction(model), (
                f"{model} lost its extinction wrapper; family is "
                f"{rm.model_type(model)!r}"
            )


def test_sncosmo_is_called_directly():
    """redback's example calls sncosmo_models itself, with no phase model.

    "For SNCOSMO model fitting, this interface is slightly different ... we can
    directly sample t0 without using a phase model."
    """
    from redback.model_library import all_models_dict

    # Wrapped twice, so an out-of-range band rejects the sample instead of
    # crashing and av_mw reddens the result, but it is redback's own function
    # underneath and no phase model stands between them.
    function = rm.observed_model(rm.SNCOSMO_MODEL)
    while hasattr(function, "__wrapped__"):
        function = function.__wrapped__
    assert function is all_models_dict[rm.SNCOSMO_MODEL]
    assert not rm.supports_extinction(rm.SNCOSMO_MODEL)
    assert rm.time_parameter(rm.SNCOSMO_MODEL) == "peak_time"
    assert rm.time_parameter("arnett") == "t0"


def test_sncosmo_ignores_t0_so_it_is_not_sampled():
    """The measured reason t0 is left out, against redback's own example.

    The example samples t0 alongside peak_time, but sncosmo_models never reads
    it: its signature is (time, redshift, model_kwargs=None, **kwargs) and t0
    lands in kwargs unused. Sampling it would be a flat direction, costing
    prior volume for nothing. If a redback release starts reading t0, this
    fails and the prior should gain it back.
    """
    from redback.transient_models.supernova_models import sncosmo_models

    times = 55570.0 + np.linspace(1, 60, 6)
    shared = dict(
        sncosmo_model="salt2",
        bands=["ztfr"] * times.size,
        output_format="magnitude",
        model_kwarg_names=["x0", "x1", "c"],
        x0=1e-7,
        x1=0.9,
        c=0.3,
        peak_time=55589.0,
    )
    a = np.asarray(sncosmo_models(time=times, redshift=0.1, t0=55560.0, **shared))
    b = np.asarray(sncosmo_models(time=times, redshift=0.1, t0=55500.0, **shared))
    assert np.allclose(a, b), "t0 now does something; sample it again"

    moved = np.asarray(
        sncosmo_models(
            time=times, redshift=0.1, t0=55560.0, **{**shared, "peak_time": 55595.0}
        )
    )
    assert not np.allclose(a, moved), "peak_time must move the light curve"


def test_sncosmo_source_parameters_follow_the_template():
    """Read from sncosmo, so any template --sncosmo-model takes is covered."""
    assert rm.sncosmo_source_parameters("v19-1993j-corr") == ("amplitude",)
    assert rm.sncosmo_source_parameters("salt2") == ("x0", "x1", "c")
    # SALT3 is not a redback model at all, only reachable here, and takes the
    # same three keys as SALT2, so salt2.prior's values apply to it.
    assert rm.sncosmo_source_parameters("salt3") == ("x0", "x1", "c")


def test_brightness_uses_the_route_redback_exercises():
    """The source parameter is sampled; use_set_peak_magnitude stays off.

    sncosmo_models also accepts ``use_set_peak_magnitude``, which would anchor
    the brightness on a peak absolute magnitude and spare us priming a raw flux
    normalisation. Nothing in redback ever sets it: the only occurrences in the
    whole package are its own definition and two docstrings, the default is
    False, and no example, notebook, test or prior file turns it on. So it is a
    provision, not a practice, and this pipeline samples what redback's own
    priors sample instead: x0/x1/c for SALT (priors/salt2.prior) and amplitude
    for the rest (priors/sn1998bw_template.prior).
    """
    for template in ("salt3", "v19-1993j-corr", "nugent-sn2p"):
        kwargs = rm.model_kwargs(rm.SNCOSMO_MODEL, ["ztfg"], sncosmo_model=template)
        assert "use_set_peak_magnitude" not in kwargs, template
        assert "peak_abs_mag" not in rm.sncosmo_keys(template), template
        # Every source parameter the template has is declared to redback.
        assert kwargs["model_kwarg_names"] == list(
            rm.sncosmo_source_parameters(template)
        )


def test_model_kwargs_declare_the_source_parameters():
    """model_kwarg_names is how redback learns which kwargs are parameters."""
    kwargs = rm.model_kwargs(rm.SNCOSMO_MODEL, ["ztfg"], sncosmo_model="v19-1993j-corr")
    assert kwargs["model_kwarg_names"] == ["amplitude"]
    assert kwargs["sncosmo_model"] == "v19-1993j-corr"
    # No phase wrapper, so no base_model.
    assert "base_model" not in kwargs

    salt = rm.model_kwargs(rm.SNCOSMO_MODEL, ["ztfg"], sncosmo_model="salt3")
    assert salt["model_kwarg_names"] == ["x0", "x1", "c"]


def test_sncosmo_priors_cover_exactly_the_templates_keys(rows):
    """Every key the template needs is in the prior, and nothing else."""
    reference = rd.first_epoch_mjd(rows)
    low, high = rm.T0_WINDOW[rm.SNCOSMO_MODEL]
    cfg = _config(model=rm.SNCOSMO_MODEL, z_known=0.0412, t0_min=low, t0_max=high)
    priors = redback_priors.build_priors(cfg, reference)

    assert set(priors) == set(rm.sncosmo_keys("v19-1993j-corr"))
    # A fixed offset window, as for every other model, NOT +/- 10 d of the
    # brightest detection: that recipe of redback's example assumes a
    # single-peaked template, and on a double-peaked IIb it pins phase zero to
    # the Ni peak and pushes the first 19 days outside the template (chi2 per
    # detection 127.3 against 4.1).
    assert priors["peak_time"].minimum == pytest.approx(reference + low)
    assert priors["peak_time"].maximum == pytest.approx(reference + high)
    # The raw normalisation, bracketing the amplitude of a real IIb at this z.
    assert priors["amplitude"].minimum < 1e-13 < priors["amplitude"].maximum
    assert "peak_abs_mag" not in priors


def test_sncosmo_salt3_uses_redbacks_own_salt2_values(rows):
    """SALT3 has no redback prior; salt2.prior's values are what it gets."""
    reference = rd.first_epoch_mjd(rows)
    priors = redback_priors.build_priors(
        _config(model=rm.SNCOSMO_MODEL, sncosmo_model="salt3"), reference
    )

    assert set(priors) == set(rm.sncosmo_keys("salt3"))
    assert priors["x0"].minimum == 1e-10 and priors["x0"].maximum == 1e-1
    assert priors["x1"].mu == 0 and priors["x1"].sigma == 1
    assert priors["c"].mu == 0 and priors["c"].sigma == 0.1


def test_sncosmo_model_evaluates_through_the_pipelines_own_call(rows):
    """The fit's own callable and kwargs must drive the template."""
    reference = rd.first_epoch_mjd(rows)
    times = reference + np.array([0.0, 5.0, 12.0, 25.0])
    bands = ["ztfg"] * times.size

    function = rm.observed_model(rm.SNCOSMO_MODEL)
    kwargs = rm.model_kwargs(rm.SNCOSMO_MODEL, bands, sncosmo_model="v19-1993j-corr")

    def magnitudes(amplitude):
        return np.asarray(
            function(
                times,
                redshift=0.0412,
                peak_time=reference + 18.0,
                amplitude=amplitude,
                ebv=0.0,
                **kwargs,
            )
        )

    mags = magnitudes(8e-14)
    assert mags.shape == times.shape
    assert np.all(np.isfinite(mags))
    # A IIb at z = 0.04 sits near 17-20 mag, not at 0 or 40.
    assert mags.min() < 22.0, mags

    # A larger amplitude must give brighter magnitudes: the parameter is read.
    brighter = magnitudes(8e-13)
    assert brighter.min() < mags.min() - 1.0


def test_sncosmo_host_extinction_follows_the_av_host_flag(rows):
    """ebv is the host E(B-V), so --sample-av-host is what drives it."""
    import bilby

    reference = rd.first_epoch_mjd(rows)

    pinned = redback_priors.build_priors(
        _config(model=rm.SNCOSMO_MODEL, sample_av_host=False), reference
    )
    assert isinstance(pinned["ebv"], bilby.core.prior.DeltaFunction)
    assert pinned["ebv"].peak == 0.0

    sampled_host = redback_priors.build_priors(
        _config(model=rm.SNCOSMO_MODEL, sample_av_host=True, av_host_max=1.0),
        reference,
    )
    assert sampled_host["ebv"].maximum == pytest.approx(1.0 / redback_priors.R_V_MW)


def test_supernova_t0_windows_follow_nmma(rows):
    """The explosion-time windows must stay tight, as NMMA's Astro-COLIBRI are.

    The first Astro-COLIBRI detection pins the epoch to about a day, so a
    window of tens of days only lets the sampler wander. NMMA's own prior
    files for these templates use Uniform(-2, +1).
    """
    for model in ("arnett", "shock_cooling", "shock_cooling_and_arnett"):
        low, high = rm.T0_WINDOW[model]
        assert (low, high) == (-2.0, 1.0), model

    # The slow transients are deliberately left wide: their rise alone lasts
    # weeks, so the explosion can genuinely predate the discovery by that much.
    for model in ("slsn", "tde_analytical", "cooling_envelope"):
        low, _ = rm.T0_WINDOW[model]
        assert low <= -100.0, model


def test_cooling_envelope_prior_keeps_the_upstream_constraint_note():
    """redback's own file documents the constraint=True requirement; keep it.

    It is the clearest statement of the bug this pipeline had, and to_file()
    drops comments, so it has to be restored deliberately.
    """
    text = (PRIOR_DIR / "cooling_envelope.prior").read_text()
    assert "constraint=True" in text
    assert "eta_min" in text and "beta_max" in text


def test_every_model_has_a_prior_file():
    for model in FORWARD_MODELS:
        assert (PRIOR_DIR / f"{model}.prior").exists(), model


def test_effective_prior_can_be_read_back(tmp_path, rows):
    """The record of a run must be replayable through --priors.

    Every bound has to be a plain Python float: bilby serialises with repr(),
    so a numpy scalar is written as "np.float64(0.33...)" and reading the file
    back raises ModuleNotFoundError: No module named 'np'. That is what
    sncosmo_max_redshift used to return, which made the effective prior of
    every sncosmo run unreadable.
    """
    import bilby

    reference = rd.first_epoch_mjd(rows)
    priors = redback_priors.build_priors(
        _config(model=rm.SNCOSMO_MODEL, z_known=None, t0_min=-10.0, t0_max=60.0),
        reference,
        bands=["ztfg", "ztfr", "atlasc", "atlaso"],
    )

    priors.to_file(outdir=str(tmp_path), label="roundtrip")
    written = tmp_path / "roundtrip.prior"
    assert "np.float64" not in written.read_text()

    back = bilby.core.prior.PriorDict(str(written))
    assert set(back) == set(priors)
    assert back["redshift"].maximum == pytest.approx(priors["redshift"].maximum)


def test_out_of_range_redshift_is_rejected_not_truncated():
    """The prior keeps its declared range; the likelihood rejects what it must.

    Past the redshift where a bandpass leaves the template's wavelength range,
    sncosmo raises instead of returning NaN. Truncating the prior to dodge that
    would remove prior volume and inflate ln Z by the log of the width removed,
    which has nothing to do with the data, so the model returns an infinite
    magnitude and the sample is rejected on its likelihood instead.
    """
    import numpy as np

    function = rm.observed_model(rm.SNCOSMO_MODEL)
    kwargs = rm.model_kwargs(
        rm.SNCOSMO_MODEL, ["ztfg"] * 3, sncosmo_model="v19-1993j-corr"
    )
    times = np.array([59430.0, 59441.0, 59450.0])
    shared = dict(peak_time=59441.0, amplitude=8e-14, ebv=0.0, **kwargs)

    inside = np.asarray(function(times, redshift=0.2, **shared))
    assert np.all(np.isfinite(inside))

    # z = 2 puts ztfg off the blue end of the template.
    outside = np.asarray(function(times, redshift=2.0, **shared))
    assert np.all(np.isinf(outside)), outside

    # And the prior itself is untouched: no truncation, so no free evidence.
    priors = redback_priors.build_priors(
        _config(model=rm.SNCOSMO_MODEL, z_known=None, t0_min=-30.0, t0_max=30.0),
        59419.226,
        bands=["ztfg", "ztfr", "atlasc", "atlaso"],
    )
    assert priors["redshift"].maximum == 3


def test_sncosmo_prior_files_split_salt_from_the_rest():
    """Two files, split on the source parameters, not one and not one each.

    A single file cannot serve every template: x0/x1/c and amplitude are
    mutually exclusive key sets, so a file carrying the SALT keys makes a v19
    fit fail with "names keys ... does not have: ['c', 'x0', 'x1']". One file
    per template would be five near-identical copies to keep in step. The split
    is read off the template, not tabulated.
    """
    import bilby

    assert not (PRIOR_DIR / f"{rm.SNCOSMO_MODEL}.prior").exists()
    assert sorted(p.stem for p in PRIOR_DIR.glob("sncosmo_*.prior")) == [
        "sncosmo_generic",
        "sncosmo_salt",
    ]

    for template, stem in (
        ("salt2", "sncosmo_salt"),
        ("salt3", "sncosmo_salt"),
        ("v19-1993j-corr", "sncosmo_generic"),
        ("nugent-sn2p", "sncosmo_generic"),
    ):
        assert rm.sncosmo_prior_stem(template) == stem, template
        keys = set(bilby.core.prior.PriorDict(str(PRIOR_DIR / f"{stem}.prior")))
        known = set(rm.sncosmo_keys(template))
        assert keys <= known, (
            f"{stem}.prior names keys {template} lacks: {keys - known}"
        )
        # peak_time is an MJD and ebv follows --sample-av-host, so neither can
        # be written down ahead of a run.
        assert "peak_time" not in keys and "ebv" not in keys, stem


def test_a_sncosmo_prior_file_wins_over_the_derived_amplitude(rows):
    """A supplied file must not be silently overwritten.

    The amplitude default is derived from the redshift prior in force, which is
    tighter than anything a static file can be. That derivation used to run
    unconditionally, after the file had been read, so the file's amplitude line
    was dead and nothing said so.
    """
    import bilby

    path = PRIOR_DIR / "sncosmo_generic.prior"
    if not path.exists():
        pytest.skip("no versioned prior for this template")
    in_file = bilby.core.prior.PriorDict(str(path))["amplitude"]
    reference = rd.first_epoch_mjd(rows)

    supplied = redback_priors.build_priors(
        _config(model=rm.SNCOSMO_MODEL, priors=path, t0_min=-10.0, t0_max=60.0),
        reference,
    )["amplitude"]
    assert supplied.minimum == in_file.minimum
    assert supplied.maximum == in_file.maximum

    derived = redback_priors.build_priors(
        _config(model=rm.SNCOSMO_MODEL, priors=None, t0_min=-10.0, t0_max=60.0),
        reference,
    )["amplitude"]
    assert derived.minimum != in_file.minimum


def test_t0_window_covers_every_model():
    """redback ships no t0 prior, so this table must cover every model."""
    for model in rm.MODELS:
        assert model in rm.T0_WINDOW
        low, high = rm.T0_WINDOW[model]
        assert low < high, model


def test_redshift_is_left_on_redbacks_own_prior_when_sampled(rows):
    """Sampling z must not narrow redback's range.

    An earlier Z_RANGE table shrank every model's redshift prior (arnett
    0.001-3 to 0.001-0.3, and so on) with nothing to justify it, which taxes
    each model with an Occam factor this pipeline invented.
    """
    from redback.priors import get_priors

    for model in FORWARD_MODELS:
        shipped = get_priors(model)["redshift"]
        cfg = _config(
            model=model,
            z_known=None,
            t0_min=rm.T0_WINDOW[model][0],
        )
        priors = redback_priors.build_priors(cfg, REFERENCE_MJD)
        assert priors["redshift"].minimum == shipped.minimum, model
        assert priors["redshift"].maximum == shipped.maximum, model


#: redback's own combined t0 + extinction model, per model family.
T0_EXTINCTION_MODEL = {
    "arnett": "t0_supernova_extinction",
    "slsn": "t0_supernova_extinction",
    "shock_cooling": "t0_shock_powered_extinction",
    "tde_analytical": "t0_tde_extinction",
    "cooling_envelope": "t0_tde_extinction",
}


def _post_t0_discrepancy(model, wrapper, times, t0):
    """Max |t0_base_model - wrapper| over post-t0 epochs, at zero extinction."""
    from redback.model_library import all_models_dict
    from redback.priors import get_priors
    from redback.utils import bands_to_frequency

    parameters = get_priors(model).sample()
    parameters["redshift"] = 0.05
    bands = ["ztfg"] * len(times)
    kwargs = {
        "output_format": "magnitude",
        "bands": bands,
        "frequency": bands_to_frequency(bands),
    }
    base = np.asarray(
        all_models_dict["t0_base_model"](
            times, t0=t0, base_model=model, **parameters, **kwargs
        ),
        dtype=float,
    )
    wrapped = np.asarray(
        all_models_dict[wrapper](
            times,
            t0=t0,
            av_host=0.0,
            av_mw=0.0,
            base_model=model,
            **parameters,
            **kwargs,
        ),
        dtype=float,
    )
    return float(np.nanmax(np.abs(base - wrapped)))


def test_t0_extinction_models_work_except_cooling_envelope():
    """Pins the real boundary of redback's extinction support.

    This corrects a claim this pipeline used to make. The ``t0_*_extinction``
    models are NOT broken in magnitude mode: compared against ``t0_base_model``
    at zero extinction, they agree to the last digit on post-t0 epochs for four
    of the five models. Only cooling_envelope diverges, because its 'spectra'
    output reports time in seconds.

    So correcting extinction on the data is a trade (one identical treatment
    for all five, hence comparable ln Z) and not a workaround. If a model moves
    from one group to the other, that trade should be reconsidered, which is
    what this test is here to announce.

    Note the comparison must exclude pre-t0 epochs: the two use different
    sentinels there (1000 vs 5000 mag), a difference with no physical meaning
    since both satisfy an upper limit and penalise a detection.
    """
    np.random.seed(0)
    t0 = 59400.0
    post_t0 = np.array([t0 + 3.0, t0 + 8.0, t0 + 15.0, t0 + 25.0])

    for model in ("arnett", "slsn", "shock_cooling", "tde_analytical"):
        discrepancy = _post_t0_discrepancy(
            model, rm.extinction_model(model), post_t0, t0
        )
        assert discrepancy < 1e-6, (
            f"{model}: t0_{rm.extinction_model(model)} no longer agrees with "
            f"t0_base_model ({discrepancy:.6f} mag)"
        )

    assert (
        _post_t0_discrepancy("cooling_envelope", "t0_tde_extinction", post_t0, t0) > 0.1
    ), "cooling_envelope's extinction wrapper now agrees; revisit deredden"


def test_t0_extinction_sentinel_differs_only_before_t0():
    """The 1000 vs 5000 mag sentinel is the only pre-t0 difference.

    Guards against mistaking it for a real disagreement, which is exactly the
    error that produced the overbroad claim above.
    """
    from redback.model_library import all_models_dict
    from redback.priors import get_priors
    from redback.utils import bands_to_frequency

    np.random.seed(0)
    t0 = 59400.0
    times = np.array([t0 - 2.0, t0 + 3.0])
    bands = ["ztfg"] * times.size
    parameters = get_priors("arnett").sample()
    parameters["redshift"] = 0.05
    kwargs = {
        "output_format": "magnitude",
        "bands": bands,
        "frequency": bands_to_frequency(bands),
    }

    base = np.asarray(
        all_models_dict["t0_base_model"](
            times, t0=t0, base_model="arnett", **parameters, **kwargs
        ),
        dtype=float,
    )
    wrapped = np.asarray(
        all_models_dict["t0_supernova_extinction"](
            times,
            t0=t0,
            av_host=0.0,
            av_mw=0.0,
            base_model="arnett",
            **parameters,
            **kwargs,
        ),
        dtype=float,
    )

    assert base[0] == 1000.0
    assert wrapped[0] == 5000.0
    assert base[1] == pytest.approx(wrapped[1])


def test_extinction_wrapper_is_an_identity_at_zero_extinction():
    """With av_host = av_mw = 0 a wrapper must reproduce its base model exactly.

    It does for every model but cooling_envelope, and that is the whole reason
    this pipeline corrects Galactic extinction on the DATA
    (redback_data.deredden) instead of wrapping the model: for
    output_format='magnitude', _evaluate_extinction_model routes through
    output_format='spectra' and passes `time_eval=time_observer_frame`
    unchanged, while cooling_envelope's own magnitude path divides it by
    day_to_s. The wrapper therefore evaluates that model on seconds where it
    means days, a ~0.9 mag error with no extinction applied at all.

    If this starts passing for cooling_envelope, upstream fixed it and the
    extinction-on-data decision can be revisited.
    """
    from redback.model_library import all_models_dict
    from redback.priors import get_priors
    from redback.utils import bands_to_frequency

    np.random.seed(0)
    times = np.linspace(5, 40, 6)
    bands = ["ztfg"] * times.size
    kwargs = {
        "output_format": "magnitude",
        "bands": bands,
        "frequency": bands_to_frequency(bands),
    }

    def discrepancy(model, wrapper):
        parameters = get_priors(model).sample()
        parameters["redshift"] = 0.05
        direct = np.asarray(
            all_models_dict[model](times, **parameters, **kwargs), dtype=float
        )
        wrapped = np.asarray(
            all_models_dict[wrapper](
                times, av_host=0.0, av_mw=0.0, base_model=model, **parameters, **kwargs
            ),
            dtype=float,
        )
        return float(np.nanmax(np.abs(direct - wrapped)))

    assert discrepancy("slsn", "extinction_with_supernova_base_model") < 1e-6
    assert discrepancy("tde_analytical", "extinction_with_tde_base_model") < 1e-6
    assert discrepancy("cooling_envelope", "extinction_with_tde_base_model") > 0.1, (
        "the cooling_envelope extinction wrapper no longer mis-scales time; "
        "revisit redback_data.deredden vs the extinction_with_* wrappers"
    )


def test_cooling_envelope_spectra_time_is_seconds():
    """Pins the upstream bug the pipeline works around.

    cooling_envelope's 'spectra' output reports time in seconds where every
    other model reports days: its magnitude path divides by day_to_s, its
    spectra path does not. This is why observed_model goes through
    t0_base_model (the magnitude path) and the extinction is applied to the
    data instead of through redback's extinction_with_* wrappers. If this test
    starts failing, upstream fixed it and that decision can be revisited.
    """
    from redback.model_library import all_models_dict
    from redback.priors import get_priors

    np.random.seed(0)
    times = np.linspace(1, 20, 5)

    days = {}
    for model in ("slsn", "tde_analytical", "shock_cooling", "cooling_envelope"):
        parameters = get_priors(model).sample()
        parameters["redshift"] = 0.05
        out = all_models_dict[model](times, **parameters, output_format="spectra")
        days[model] = float(np.max(np.asarray(out.time, dtype=float)))

    # Relative, not absolute: the numbers scale with the sampled parameters, so
    # a fixed ceiling only tests the draw. What the bug actually says is that
    # cooling_envelope's figure is ~day_to_s times larger than it should be, so
    # compare it with the models that report days.
    in_days = max(days["slsn"], days["tde_analytical"], days["shock_cooling"])
    ratio = days["cooling_envelope"] / in_days

    assert ratio > 100, (
        "cooling_envelope's spectra time is no longer out of scale with the "
        f"models that report days (ratio {ratio:.1f}); upstream may have fixed "
        "it, so revisit EXTINCTION_MODEL and the no-extinction fallback"
    )
    # And that dividing by day_to_s brings it back into their range, which is
    # what identifies the unit as seconds rather than merely "large".
    from redback.constants import day_to_s

    assert days["cooling_envelope"] / day_to_s < in_days


# --------------------------------------------------------------------------
# Injection-recovery
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_injection_recovery_slsn(tmp_path):
    """Recover a known slsn: the truth must land inside the 5-95% interval."""
    import bilby
    import redback
    from redback.utils import bands_to_frequency

    truth = {
        "redshift": 0.1,
        "p0": 2.0,
        "bp": 1.0,
        "mass_ns": 1.4,
        "theta_pb": 0.5,
        "mej": 5.0,
        "vej": 10000.0,
        "kappa": 0.1,
        "kappa_gamma": 0.03,
        "temperature_floor": 6000.0,
    }
    t0 = 59400.0
    rng = np.random.default_rng(7)

    epochs = np.sort(rng.uniform(2.0, 60.0, 25))
    times = np.concatenate(([t0 - 6.0, t0 - 4.0, t0 - 2.0], t0 + epochs))
    bands = ["ztfg", "ztfr"] * (times.size // 2) + ["ztfg"] * (times.size % 2)

    model = rm.observed_model("slsn")
    clean = np.asarray(
        model(
            times,
            t0=t0,
            base_model="slsn",
            output_format="magnitude",
            bands=bands,
            frequency=bands_to_frequency(bands),
            **truth,
        )
    )
    noisy = clean + rng.normal(0.0, 0.05, clean.size)

    rows = [
        {
            "mjd": float(time),
            "band": band,
            "mag": float(magnitude) if time >= t0 else 21.0,
            "mag_err": 0.05 if time >= t0 else float("nan"),
            "detected": bool(time >= t0),
        }
        for time, band, magnitude in zip(times, bands, noisy)
    ]

    transient = rd.build_transient(rows, "injection", error_budget=0.05)

    priors = bilby.core.prior.PriorDict(
        dictionary=dict(__import__("redback.priors", fromlist=["x"]).get_priors("slsn"))
    )
    priors["redshift"] = bilby.core.prior.DeltaFunction(
        truth["redshift"], name="redshift"
    )
    priors["kappa"] = bilby.core.prior.DeltaFunction(truth["kappa"], name="kappa")
    priors["t0"] = bilby.core.prior.Uniform(t0 - 20.0, t0 + 2.0, name="t0")

    result = redback.fit_model(
        transient=transient,
        model=rm.observed_model("slsn"),
        prior=priors,
        model_kwargs={
            "output_format": "magnitude",
            "base_model": "slsn",
            "bands": transient.filtered_bands,
            "frequency": bands_to_frequency(list(transient.filtered_bands)),
        },
        sampler="dynesty",
        nlive=200,
        outdir=str(tmp_path),
        label="injection",
        plot=False,
        resume=False,
        clean=True,
        sample="rslice",
    )

    assert np.isfinite(result.log_evidence)
    for parameter in ("t0", "p0", "mej"):
        low, high = np.percentile(result.posterior[parameter].to_numpy(), [5, 95])
        expected = t0 if parameter == "t0" else truth[parameter]
        assert low <= expected <= high, (
            f"{parameter}: {expected} outside [{low}, {high}]"
        )


def test_every_likelihood_resolves_to_a_redback_class():
    """``--likelihood`` must name classes redback actually ships.

    The registry carries the sigma formula in prose; if a future redback
    renames a class this fails here instead of at the first likelihood call,
    hours into a fit.
    """
    import redback_models as rm

    for name, spec in rm.LIKELIHOODS.items():
        cls = rm.likelihood_class(name)
        assert cls.__name__ == spec["cls"]
        assert "sigma" in spec and "note" in spec

    assert rm.DEFAULT_LIKELIHOOD in rm.LIKELIHOODS


def test_default_likelihood_leaves_the_choice_to_redback():
    """The default must return None, which is ``_fit_optical_transient``'s own.

    Passing a class built here would duplicate redback's logic and could drift
    from it; returning None keeps that path byte-for-byte redback's.
    """
    import fit_redback

    assert fit_redback.build_likelihood(_config(), object(), {}) is None


def test_budget_likelihoods_refuse_to_drop_upper_limits():
    """A budget likelihood has no survival term, so limits must not vanish.

    Silently fitting detections only would change the t0 posterior and the
    ln Z without saying so, which is the one failure mode worth a hard stop.
    """
    import fit_redback
    import numpy as np
    import pytest

    class _WithLimits:
        def get_filtered_data_with_limits(self):
            n = 5
            detections = np.array([True, True, False, True, False])
            return (
                np.arange(n, dtype=float),
                None,
                np.full(n, 20.0),
                np.full(n, 0.1),
                detections,
            )

    with pytest.raises(SystemExit, match="2 upper limits"):
        fit_redback.build_likelihood(
            _config(likelihood="quadrature"), _WithLimits(), {}
        )


def test_budget_likelihoods_sample_sigma():
    """The budget classes read a parameter named ``sigma``.

    Without it in the prior bilby never varies it and the budget is silently
    pinned at whatever the class defaults to, which would look like a working
    fit.
    """
    from redback_priors import build_priors

    plain = build_priors(_config(), 59419.0)
    assert "sigma" not in plain

    for name in ("quadrature", "systematic"):
        priors = build_priors(_config(likelihood=name), 59419.0)
        assert priors["sigma"].maximum == 0.5
        assert priors["sigma"].minimum == 0.0


def test_effective_errors_widen_only_for_budget_likelihoods():
    """The residual strip must use the same sigma the likelihood did."""
    import types

    import numpy as np
    import pandas
    import plot_redback

    errors = np.array([0.1, 0.1])
    draws = np.array([[20.0, 20.0], [20.0, 20.0]])
    result = types.SimpleNamespace(posterior=pandas.DataFrame({"sigma": [0.2, 0.2]}))

    unchanged = plot_redback.effective_errors(_config(), result, errors, draws)
    assert np.allclose(unchanged, errors)

    widened = plot_redback.effective_errors(
        _config(likelihood="quadrature"), result, errors, draws
    )
    assert np.allclose(widened, np.sqrt(0.1**2 + 0.2**2))

    scaled = plot_redback.effective_errors(
        _config(likelihood="systematic"), result, errors, draws
    )
    assert np.allclose(scaled, np.sqrt(0.1**2 + (20.0 * 0.2) ** 2))


def test_prior_file_extinction_wins_over_the_flags(tmp_path):
    """An E(B-V) written in a --priors file must not be silently overwritten.

    The flags derive a bound from a dust map or from --av-host-max; a line in
    the file is a value the caller already measured. Overwriting it without a
    word was the behaviour before, and it changed the prior volume, so the
    ln Z too.
    """
    import pytest
    from redback_priors import build_priors

    base = dict(
        model="sncosmo_models",
        sncosmo_model="v19-1993j-corr",
        sample_av_host=True,
        av_host_max=1.0,
        z_known=None,
    )

    derived = build_priors(_config(**base), 59419.0, bands=["ztfg"])
    assert derived["ebv"].maximum == pytest.approx(1.0 / 3.1)

    path = tmp_path / "with_ebv.prior"
    path.write_text(
        "redshift = Uniform(minimum=0.01, maximum=3, name='redshift')\n"
        "amplitude = LogUniform(minimum=1e-22, maximum=1e-2, name='amplitude')\n"
        "ebv = Uniform(minimum=0, maximum=0.08, name='ebv')\n"
    )
    kept = build_priors(_config(priors=path, **base), 59419.0, bands=["ztfg"])
    assert kept["ebv"].maximum == 0.08


def test_galactic_extinction_reddens_the_template_per_band():
    """av_mw must add the fitzpatrick99 A_lambda of each band, nothing else.

    redback leaves this path with an inert 'mw' effect: it registers F99Dust
    but never sets mwebv, and the parameter cannot be supplied because
    model.update() runs before add_effect. Without this wrapper the Galactic
    reddening is silently absorbed by the host 'ebv' key and reported as host
    extinction.
    """
    import extinction
    import numpy as np

    model = rm.observed_model(rm.SNCOSMO_MODEL)
    bands = ["ztfg", "ztfr", "atlasc", "atlaso"]
    shared = dict(
        redshift=0.0412,
        peak_time=59419.0,
        amplitude=1e-13,
        ebv=0.0,
        **rm.model_kwargs(rm.SNCOSMO_MODEL, bands, sncosmo_model="v19-1993j-corr"),
    )
    times = np.full(len(bands), 59425.0)

    bare = model(times, av_mw=0.0, **shared)
    av_mw = 3.1 * 0.0637
    reddened = model(times, av_mw=av_mw, **shared)

    waves = np.array([rm._wave_eff(b) for b in bands])
    expected = extinction.fitzpatrick99(waves, av_mw, 3.1)
    assert np.allclose(reddened - bare, expected)

    # Monotonic with wavelength, and the right order of magnitude: 0.233 mag in
    # g for the SN2021ugl line of sight.
    assert expected[0] == pytest.approx(0.233, abs=5e-3)
    assert expected[0] > expected[1]


def test_av_mw_never_reaches_model_update():
    """sncosmo would reject 'av_mw': it is not a template parameter.

    model_kwarg_names is what redback feeds to model.update(), so av_mw must
    stay out of it and be consumed by the wrapper instead.
    """
    names = rm.model_kwargs(rm.SNCOSMO_MODEL, ["ztfg"], sncosmo_model="v19-1993j-corr")[
        "model_kwarg_names"
    ]
    assert "av_mw" not in names
    assert "av_mw" in rm.sncosmo_keys("v19-1993j-corr")


def test_t0_window_comes_from_the_data_not_a_table():
    """The window must adapt to each file's cadence, with no per-target tuning.

    A fixed (-2, +1) is 3 days wide wherever it is applied, but the gap from
    the first epoch to the first detection is 3.99 d on SN2021ugl and 6.04 d
    on SN2020oi. On the second the fixed window sits entirely before the deep
    pre-explosion limit, so it places the explosion where the data exclude it.
    """
    reference, detection = 58849.503, 58855.542
    assert rm.t0_window("arnett", reference, detection) == (0.0, detection - reference)

    # Upper bound is the first detection: t0_base_model returns 1000 mag
    # before t0, so any later value is dead prior volume and only taxes ln Z.
    low, high = rm.t0_window("shock_cooling_and_arnett", reference, detection)
    assert reference + high == pytest.approx(detection)
    assert low == 0.0


def test_t0_window_keeps_the_table_for_peak_time_models():
    """sncosmo samples peak_time, maximum light, which the first detection
    does not bound: on a double-peaked IIb it belongs before it."""
    assert (
        rm.t0_window(rm.SNCOSMO_MODEL, 58849.503, 58855.542)
        == rm.T0_WINDOW[rm.SNCOSMO_MODEL]
    )


def test_t0_window_falls_back_without_epochs():
    """No epochs, or a file starting on its first detection, keeps the table."""
    assert rm.t0_window("arnett") == rm.T0_WINDOW["arnett"]
    # Nothing before the transient: the model's own slack, capped at the
    # detection so t0 cannot land after the object was seen.
    assert rm.t0_window("arnett", 100.0, 100.0) == (rm.T0_WINDOW["arnett"][0], 0.0)


@pytest.mark.parametrize("name", ["SN2021ugl"])
def test_t0_window_never_allows_t0_after_the_first_detection(name):
    """The one invariant worth a guard: a transient explodes before it is seen."""
    rows = rd.read_photometry(DATA / "processed" / f"{name}.dat")
    reference = rd.first_epoch_mjd(rows)
    detection = rd.first_detection_mjd(rows)
    for model in rm.MODELS:
        if model == rm.SNCOSMO_MODEL:
            continue
        _, high = rm.t0_window(model, reference, detection)
        assert reference + high <= detection + 1e-9, model


def test_t0_never_starts_after_the_first_fitted_detection(tmp_path):
    """--detections-only must not shift the window past the first detection.

    The window is a pair of offsets from the first row of the data actually
    fitted. Computing them on the whole file while the fit sees only the
    detections moved t0 to [first detection, +6.04 d] on SN2020oi, so
    t0_base_model returned its 1000 mag sentinel for every earlier epoch:
    ln Z went to -4.9e7 and the sampled systematic railed at its ceiling.
    """
    import argparse

    import fit_redback

    base = dict(
        object=None,
        data=DATAFILE,
        outdir=tmp_path,
        model="arnett",
        priors=None,
        kappa=None,
        z_known=None,
        z_phot=None,
        z_phot_err=None,
        ra=None,
        dec=None,
        ebv_mw=0.0637,
        sample_av_host=False,
        av_host_max=1.0,
        t0_min=None,
        t0_max=None,
        sncosmo_model="v19-1993j-corr",
        likelihood="upper-limits",
        sigma_sys_max=0.5,
        plot_theme="print",
        data_tmin=None,
        data_tmax=None,
        upper_limit_sigma=3.0,
        nlive=100,
        sampler="dynesty",
        no_constraint=False,
        seed=42,
    )
    rows = rd.read_photometry(DATAFILE)
    detection = rd.first_detection_mjd(rows)

    for detections_only in (False, True):
        args = argparse.Namespace(
            **{**base, "object": "SN2021ugl", "detections_only": detections_only}
        )
        cfg = fit_redback.resolve_config(args)
        kept = [r for r in rows if r["detected"]] if detections_only else rows
        reference = rd.first_epoch_mjd(kept)
        assert reference + cfg.t0_max <= detection + 1e-9, (
            f"detections_only={detections_only}: t0 may start after the first "
            f"detection, where the model is the 1000 mag sentinel"
        )


def test_a_railed_systematic_is_reported():
    """A sigma clipped by its ceiling must not pass silently.

    It inflates every chi-square and narrows M_Ni's interval while nothing in
    the output looks wrong, which is how two SN2020oi runs went unnoticed.
    """
    import types

    import fit_redback
    import numpy as np
    import pandas

    cfg = _config(likelihood="quadrature", sigma_sys_max=0.25)

    railed = types.SimpleNamespace(
        posterior=pandas.DataFrame({"sigma": np.linspace(0.238, 0.25, 500)})
    )
    assert fit_redback.warn_if_sigma_railed(railed, cfg) is True

    healthy = types.SimpleNamespace(
        posterior=pandas.DataFrame({"sigma": np.linspace(0.02, 0.12, 500)})
    )
    assert fit_redback.warn_if_sigma_railed(healthy, cfg) is False

    # The default likelihood samples no sigma at all.
    none = types.SimpleNamespace(posterior=pandas.DataFrame({"mej": [1.0, 2.0]}))
    assert fit_redback.warn_if_sigma_railed(none, _config()) is False
