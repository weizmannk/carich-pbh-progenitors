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

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "fitting" / "redback-fit"))

import redback_data as rd  # noqa: E402
import redback_models as rm  # noqa: E402
import redback_priors  # noqa: E402

CONSTRAINED_MODELS = ("slsn", "tde_analytical", "cooling_envelope")
DATAFILE = REPO_ROOT / "data" / "processed" / "SN2021ugl.dat"
PRIOR_DIR = REPO_ROOT / "priors_redback"
REFERENCE_MJD = rd.first_detection_mjd(rd.read_photometry(DATAFILE))


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
        data_tmin=None,
        data_tmax=None,
        upper_limit_sigma=5.0,
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


@pytest.mark.parametrize("model", list(rm.MODELS))
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
        rd.build_transient(swapped, "X", upper_limit_sigma=5.0)


def test_no_detection_rejected(rows):
    limits_only = [{**row, "detected": False} for row in rows]
    with pytest.raises(ValueError, match="no detection"):
        rd.build_transient(limits_only, "X", upper_limit_sigma=5.0)


def test_measured_errors_pass_through_untouched(rows):
    """No systematic floor is folded into the data, as in redback's own loader.

    An earlier version added NMMA's sigma_eb in quadrature, which compressed the
    relative weighting of this data set by a factor of 14: the best detection
    (0.043 mag) outweighs the worst (0.207) by 23x on the measured errors, but
    only 1.6x once a 0.25 mag floor dominates both.
    """
    transient = rd.build_transient(rows, "SN2021ugl", upper_limit_sigma=5.0)
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
    transient = rd.build_transient(rows, "SN2021ugl", upper_limit_sigma=5.0)
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
    transient = rd.build_transient(rows, "SN2021ugl", upper_limit_sigma=5.0)
    assert transient.upper_limit_sigma == 5.0
    assert transient.has_upper_limits
    # The survival width redback derives from it.
    survival_width = (2.5 / np.log(10.0)) / transient.upper_limit_sigma
    assert survival_width == pytest.approx(0.2171, abs=1e-4)


def test_transient_is_on_mjd_scale(rows):
    """The likelihood reads `time_mjd`, so Transient.x must be MJD."""
    transient = rd.build_transient(rows, "SN2021ugl", upper_limit_sigma=5.0)
    assert transient.x.min() > 50000.0, "x should be MJD, not a phase"
    assert np.all(np.diff(transient.x) >= 0)
    assert transient.first_detection_mjd == pytest.approx(rd.first_detection_mjd(rows))


def test_data_window_relative_to_first_detection(rows):
    """data_tmin may be negative, to keep the limits that constrain t0."""
    reference = rd.first_detection_mjd(rows)
    kept = rd.build_transient(rows, "X", upper_limit_sigma=5.0, data_tmin=0.0)
    assert kept.x.min() >= reference

    with_limits = rd.build_transient(rows, "X", upper_limit_sigma=5.0, data_tmin=-10.0)
    assert with_limits.x.min() < reference, "pre-detection limits should survive"
    assert len(with_limits.x) > len(kept.x)


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

    reference = rd.first_detection_mjd(rows)
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


def test_shock_cooling_prior_pins_nn_delta():
    """Piro, Haynie & Yao (2021): n ~ 10, delta ~ 1.1 (Chevalier & Soker 1989)."""
    import bilby

    priors = bilby.core.prior.PriorDict(str(PRIOR_DIR / "shock_cooling.prior"))
    assert isinstance(priors["nn"], bilby.core.prior.DeltaFunction)
    assert priors["nn"].peak == 10
    assert isinstance(priors["delta"], bilby.core.prior.DeltaFunction)
    assert priors["delta"].peak == 1.1


#: Keys priors_redback deliberately pins where redback samples them, with the
#: reason. Everything else must match redback exactly.
DELIBERATE_OVERRIDES = {
    ("shock_cooling", "nn"),
    ("shock_cooling", "delta"),
    ("shock_cooling_and_arnett", "nn"),
    ("shock_cooling_and_arnett", "delta"),
}


@pytest.mark.parametrize("model", list(rm.MODELS))
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
            assert isinstance(versioned[key], bilby.core.prior.DeltaFunction), (
                f"{key} is listed as a deliberate override but is not pinned"
            )
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
    for model in rm.MODELS:
        text = (PRIOR_DIR / f"{model}.prior").read_text()
        assert text.lstrip().startswith("#"), f"{model}.prior has no header"
        assert "get_priors" in text, f"{model}.prior does not name its source"


def test_cooling_envelope_prior_keeps_the_upstream_constraint_note():
    """redback's own file documents the constraint=True requirement; keep it.

    It is the clearest statement of the bug this pipeline had, and to_file()
    drops comments, so it has to be restored deliberately.
    """
    text = (PRIOR_DIR / "cooling_envelope.prior").read_text()
    assert "constraint=True" in text
    assert "eta_min" in text and "beta_max" in text


def test_every_model_has_a_prior_file():
    for model in rm.MODELS:
        assert (PRIOR_DIR / f"{model}.prior").exists(), model


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

    for model in rm.MODELS:
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
