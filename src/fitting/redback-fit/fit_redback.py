"""Fit one redback model to one transient. CLI counterpart of NMMA's lightcurve-analysis.

Usage:
    python fit_redback.py --object SN2021ugl --model slsn --outdir results/

What is sampled
---------------
``t0`` is sampled as an absolute MJD, redback's own parameterisation, which
``t0_base_model`` requires, inside a window anchored on the first detection
(``redback_models.T0_WINDOW``), the way redback's own examples anchor theirs.
Fixing it would understate every other uncertainty, and the pre-explosion upper
limits are what constrain it, which is why ``--data-tmin`` defaults to keeping
them.

``redshift`` is fixed (``--z-known``), Gaussian (``--z-phot``), or sampled on
redback's own prior when neither is given, optional, as on the NMMA side.

``kappa`` is sampled unless ``--kappa`` pins it. Pinning is the honest choice
when the data have no 56-Co tail: the diffusion timescale then constrains only
``mej * kappa / vej``, so a free kappa spreads an unconstrained direction into
every other posterior. Run the same object at 0.05, 0.07 and 0.10 to expose
that dependence rather than hiding it inside one error bar.

What is measured
----------------
``M_Ni = f_nickel * mej`` is reported for the models that expose a nickel
fraction (``arnett``), propagated from the joint posterior rather than from the
two marginals, since the parameters are correlated.

``M_WD`` is deliberately absent. No ejecta model carries the progenitor mass;
converting M_ej to M_WD assumes complete disruption and belongs in the paper's
discussion, not in a fit result.

Galactic extinction is **sampled in the forward model**, through redback's own
``t0_<family>_extinction`` models, with ``av_mw`` bounded by the dust map
(``Uniform(0, R_V x E(B-V)_map)``) exactly as NMMA bounds ``--Ebv-max``. The
observed magnitudes are never altered: dereddening them would impose the map
value as exact, and on SN2021ugl that value (E(B-V) = 0.0637) is 5.4x what
NMMA's free parameter settles at (0.0118), a 0.19 mag one-directional
over-correction in g, straight onto M_Ni.

``cooling_envelope`` is the exception: its ``spectra`` output reports time in
seconds where every other model reports days, and those wrappers route
magnitude output through ``spectra``, so it would be evaluated on the wrong
time axis (1.55 mag off with zero extinction). It therefore carries **no**
extinction at all, which the summary records, and its ln Z is not comparable
with the others', the same kind of exclusion the NMMA service paper already
makes for Piro2021 outside its domain of validity.

See tests/test_redback_pipeline.py::test_t0_extinction_models_work_except_cooling_envelope.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import plot_redback
import redback
from model_descriptions import describe
from redback_data import build_transient, first_epoch_mjd, read_photometry
from redback_models import (
    DEFAULT_LIKELIHOOD,
    LIKELIHOODS,
    MODELS,
    SNCOSMO_MODEL,
    T0_WINDOW,
    likelihood_class,
    model_kwargs,
    observed_model,
    sncosmo_prior_stem,
    time_parameter,
)
from redback_priors import build_priors, extinction_summary

logger = logging.getLogger(__name__)


# Parameters fixed by the caller or by a versioned prior file rather than
# sampled, with the reason, for the summary.
FIXED_BY_DESIGN = {
    "kappa": "degenerate with mej without a 56-Co tail",
    "nn": "fixed at the Piro, Haynie & Yao (2021) value; redback samples it",
    "delta": "fixed at the Piro, Haynie & Yao (2021) value; redback samples it",
}


@dataclass
class RunConfig:
    """Everything one fit needs, resolved from the command line."""

    object_name: str
    datafile: Path
    outdir: Path
    model: str
    priors: Path | None
    kappa: float | None
    z_known: float | None
    z_phot: float | None
    z_phot_err: float | None
    ebv_mw: float
    ebv_source: str
    sample_av_host: bool
    av_host_max: float
    t0_min: float
    t0_max: float
    sncosmo_model: str
    likelihood: str
    sigma_sys_max: float
    detections_only: bool
    plot_theme: str
    data_tmin: float | None
    data_tmax: float | None
    upper_limit_sigma: float
    nlive: int
    sampler: str
    no_constraint: bool
    seed: int

    @property
    def label(self) -> str:
        """``{model}_{object}``, the label NMMA's comparison tools expect."""
        return f"{self.model}_{self.object_name}"

    @property
    def redshift_mode(self) -> str:
        if self.z_known is not None:
            return "fixed"
        if self.z_phot is not None:
            return "photometric"
        return "sampled"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--object", required=True, help="Object name (e.g. SN2021ugl, ZTF20abaunmw)"
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=tuple(MODELS),
        help="Redback model to fit",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Photometry file. Default: data/processed/<object>.dat",
    )
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument(
        "--priors",
        type=Path,
        default=None,
        help="bilby prior file overriding redback's bundled prior key by key. "
        "Default: priors_redback/<model>.prior if it exists",
    )

    redshift = parser.add_mutually_exclusive_group()
    redshift.add_argument(
        "--z-known",
        type=float,
        default=None,
        dest="z_known",
        help="Spectroscopic redshift; fixes it instead of sampling. Optional",
    )
    redshift.add_argument(
        "--z-phot",
        type=float,
        default=None,
        dest="z_phot",
        help="Photometric redshift; sampled as a truncated Gaussian (needs --z-phot-err)",
    )
    parser.add_argument(
        "--z-phot-err",
        type=float,
        default=None,
        dest="z_phot_err",
        help="Sigma on --z-phot",
    )

    parser.add_argument(
        "--kappa",
        type=float,
        default=None,
        help="Pin the opacity (cm^2/g) instead of sampling it: 0.05, 0.07 or 0.10",
    )
    parser.add_argument(
        "--ra", type=float, default=None, help="Right ascension, deg (ICRS)"
    )
    parser.add_argument(
        "--dec", type=float, default=None, help="Declination, deg (ICRS)"
    )
    parser.add_argument(
        "--ebv-mw",
        type=float,
        default=None,
        dest="ebv_mw",
        help="Milky Way E(B-V); bounds the sampled av_mw at R_V x E(B-V). "
        "Takes precedence over the --ra/--dec map query",
    )
    parser.add_argument(
        "--sample-av-host",
        action="store_true",
        dest="sample_av_host",
        help="Also sample host extinction. Off by default: no dust map bounds "
        "it, and it is degenerate with the ejecta parameters",
    )
    parser.add_argument(
        "--av-host-max",
        type=float,
        default=1.0,
        dest="av_host_max",
        help="Upper bound on av_host when --sample-av-host is given",
    )

    parser.add_argument(
        "--t0-min",
        type=float,
        default=None,
        dest="t0_min",
        help="Earliest t0, days relative to the first detection (negative = "
        "before it). Default: T0_WINDOW",
    )
    parser.add_argument(
        "--t0-max",
        type=float,
        default=None,
        dest="t0_max",
        help="Latest t0, same units",
    )
    parser.add_argument(
        "--likelihood",
        choices=tuple(LIKELIHOODS),
        default=DEFAULT_LIKELIHOOD,
        help="Which redback likelihood to use. upper-limits keeps the "
        "non-detections through a survival term and carries no systematic "
        "budget; quadrature adds a sampled sigma in quadrature, the formula "
        "NMMA uses behind --em-error-budget, and drops the limits; systematic "
        "scales that budget by the model value, which suits flux not "
        "magnitudes. No redback likelihood does both",
    )
    parser.add_argument(
        "--sigma-sys-max",
        type=float,
        default=0.5,
        dest="sigma_sys_max",
        help="Upper bound on the sampled sigma, for the budget likelihoods. "
        "Pin it with a DeltaFunction on 'sigma' in a --priors file to "
        "reproduce NMMA's fixed --em-error-budget",
    )
    parser.add_argument(
        "--detections-only",
        action="store_true",
        dest="detections_only",
        help="Drop the upper limits before fitting. Required by the budget "
        "likelihoods, which have no survival term",
    )
    parser.add_argument(
        "--plot-theme",
        choices=tuple(plot_redback.THEMES),
        default=plot_redback.DEFAULT_THEME,
        dest="plot_theme",
        help="Figure palette. Only 'print', a plain figure for a paper, ships "
        "with this repository; others appear when a local palette module is "
        "present",
    )
    parser.add_argument(
        "--sncosmo-model",
        default="v19-1993j-corr",
        dest="sncosmo_model",
        help="sncosmo template name, used only by --model sncosmo_models. Any "
        "source sncosmo can load, e.g. v19-1993j-corr, nugent-sn2p, salt3",
    )
    parser.add_argument(
        "--data-tmin",
        type=float,
        default=None,
        dest="data_tmin",
        help="Keep photometry from this many days relative to the first detection "
        "(negative keeps the pre-explosion limits, which constrain t0)",
    )
    parser.add_argument(
        "--data-tmax",
        type=float,
        default=None,
        dest="data_tmax",
        help="Upper end of that window",
    )

    parser.add_argument(
        "--upper-limit-sigma",
        type=float,
        default=3.0,
        dest="upper_limit_sigma",
        help="Sigma level of the non-detections. redback's own default, 3.0, "
        "which sets the survival width to (2.5/ln10)/3 = 0.362 mag. Raise it "
        "only if the file's limits are known to be at another level; this one "
        "records none",
    )
    parser.add_argument("--nlive", type=int, default=2048)
    parser.add_argument(
        "--sampler",
        default="pymultinest",
        choices=("pymultinest", "dynesty"),
        help="pymultinest needs libmultinest on the library path "
        "(make setup-multinest); dynesty is pure Python",
    )
    parser.add_argument(
        "--no-constraint",
        action="store_true",
        dest="no_constraint",
        help="Drop redback's physical constraint priors for this model",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def repo_root() -> Path:
    """Repository root, from this file's location."""
    return Path(__file__).resolve().parents[3]


def resolve_extinction(args: argparse.Namespace) -> tuple[float, str]:
    """Milky Way E(B-V) and where it came from.

    ``--ebv-mw`` wins; otherwise the Planck GNILC map is queried at ``--ra`` /
    ``--dec``. Both are plain arguments, as on the NMMA side: the batch layer
    (``condor/submit_batch.py``) is what reads ``config/targets.csv``. A failure
    is not
    fatal, the fit proceeds with no correction and says so, because an
    unreachable dust map should not block a run on a high-latitude target.
    """
    if args.ebv_mw is not None:
        return float(args.ebv_mw), "user"

    coordinates = (
        (args.ra, args.dec) if args.ra is not None and args.dec is not None else None
    )
    if coordinates is None:
        logger.warning(
            "No --ebv-mw and no --ra/--dec for %s: av_mw will be pinned at 0, "
            "so no Galactic extinction is marginalised over",
            args.object,
        )
        return 0.0, "none"

    sys.path.insert(0, str(repo_root() / "src" / "ingest"))
    try:
        from mw_extinction import get_planck_ebv

        return float(get_planck_ebv(*coordinates)), "planck_gnilc"
    except Exception as exc:
        logger.warning("Planck GNILC query failed (%s): av_mw will be pinned at 0", exc)
        return 0.0, "none"


def resolve_config(args: argparse.Namespace) -> RunConfig:
    """Fill in every default that depends on the repository layout."""
    datafile = args.data or repo_root() / "data" / "processed" / f"{args.object}.dat"
    if not datafile.exists():
        raise SystemExit(f"Photometry not found: {datafile}")

    if args.z_phot is not None and args.z_phot_err is None:
        raise SystemExit("--z-phot needs --z-phot-err")

    # The redshift is optional, exactly as on the NMMA side: give --z-known to
    # fix it, --z-phot for a photometric one, or neither and redback's own
    # prior is sampled. Astro-COLIBRI supplies one for some objects and not for
    # others, so no value is assumed here.
    if args.z_known is None and args.z_phot is None:
        logger.info(
            "No redshift given for %s: sampling it on redback's own prior",
            args.object,
        )

    priors = args.priors
    if priors is None:
        # The sncosmo path resolves by TEMPLATE, not by model name: one file
        # cannot serve them all, since x0/x1/c and amplitude are mutually
        # exclusive key sets.
        stem = (
            sncosmo_prior_stem(args.sncosmo_model)
            if args.model == SNCOSMO_MODEL
            else args.model
        )
        default_prior = repo_root() / "priors_redback" / f"{stem}.prior"
        priors = default_prior if default_prior.exists() else None

    window = T0_WINDOW[args.model]
    ebv_mw, ebv_source = resolve_extinction(args)

    return RunConfig(
        object_name=args.object,
        datafile=datafile,
        outdir=args.outdir,
        model=args.model,
        priors=priors,
        kappa=args.kappa,
        z_known=args.z_known,
        z_phot=args.z_phot,
        z_phot_err=args.z_phot_err,
        ebv_mw=ebv_mw,
        ebv_source=ebv_source,
        sample_av_host=args.sample_av_host,
        av_host_max=args.av_host_max,
        t0_min=window[0] if args.t0_min is None else args.t0_min,
        t0_max=window[1] if args.t0_max is None else args.t0_max,
        sncosmo_model=args.sncosmo_model,
        likelihood=args.likelihood,
        sigma_sys_max=args.sigma_sys_max,
        detections_only=args.detections_only,
        plot_theme=args.plot_theme,
        data_tmin=args.data_tmin,
        data_tmax=args.data_tmax,
        upper_limit_sigma=args.upper_limit_sigma,
        nlive=args.nlive,
        sampler=args.sampler,
        no_constraint=args.no_constraint,
        seed=args.seed,
    )


def build_likelihood(cfg, transient, model_kwargs_dict):
    """The redback likelihood ``--likelihood`` names, or None for its default.

    Returns None for ``upper-limits``, which is what ``fit_model`` picks on its
    own when non-detections are present, so that path stays exactly as redback
    wrote it.

    The two budget variants add a sampled ``sigma`` on top of the photometric
    errors, which is the same construction NMMA applies behind
    ``--em-error-budget``, except that NMMA fixes the value and here the data
    estimate it. Neither has a survival term, so they refuse to run on data
    that still carries limits rather than dropping them in silence.

    Raises
    ------
    SystemExit
        If a budget likelihood is asked for while upper limits are present.
    """
    if cfg.likelihood == DEFAULT_LIKELIHOOD:
        return None

    # The same accessor redback's own `_fit_optical_transient` calls, so the
    # likelihood sees the bands the fit kept and not the whole file.
    x, _, y, y_err, detections = transient.get_filtered_data_with_limits()

    if detections is not None and not np.all(detections):
        raise SystemExit(
            f"--likelihood {cfg.likelihood} has no survival term and "
            f"{cfg.object_name} still carries {int(np.sum(~detections))} upper "
            "limits. Add --detections-only to drop them deliberately, or keep "
            f"--likelihood {DEFAULT_LIKELIHOOD}."
        )

    spec = LIKELIHOODS[cfg.likelihood]
    logger.info("likelihood %s: sigma = %s", spec["cls"], spec["sigma"])
    return likelihood_class(cfg.likelihood)(
        x=np.asarray(x, dtype=float),
        y=np.asarray(y, dtype=float),
        sigma_i=np.asarray(y_err, dtype=float),
        function=observed_model(cfg.model),
        kwargs=model_kwargs_dict,
    )


def nickel_mass(posterior) -> dict:
    """M_Ni and its credible interval, from the joint posterior.

    Computed sample by sample so the correlation between ``f_nickel`` and
    ``mej`` is carried through; multiplying the two medians would not give
    the median of the product.
    """
    if "f_nickel" not in posterior or "mej" not in posterior:
        return {"error": "model exposes no f_nickel; M_Ni is not defined"}

    samples = posterior["f_nickel"].to_numpy() * posterior["mej"].to_numpy()
    low, median, high = np.percentile(samples, [16, 50, 84])
    return {
        "median": float(median),
        "minus": float(median - low),
        "plus": float(high - median),
        "unit": "Msun",
    }


def summarise(result, cfg: RunConfig, transient, priors) -> dict:
    """The JSON record of one fit."""
    import bilby

    posterior = result.posterior
    parameters: dict[str, dict] = {}
    for column in posterior.columns:
        if column in ("log_likelihood", "log_prior"):
            continue
        if not np.issubdtype(posterior[column].dtype, np.number):
            continue
        values = posterior[column].to_numpy()
        if np.allclose(values, values[0]):
            parameters[column] = {
                "value": float(values[0]),
                "fixed": True,
                "reason": FIXED_BY_DESIGN.get(column, "fixed by the prior"),
            }
            continue
        low, median, high = np.percentile(values, [16, 50, 84])
        parameters[column] = {
            "median": float(median),
            "minus": float(median - low),
            "plus": float(high - median),
            "fixed": False,
        }

    detections = transient.detections
    n_total = len(np.atleast_1d(transient.y))
    n_detections = (
        n_total
        if detections is None
        else int(np.sum(np.asarray(detections, dtype=bool)))
    )

    # t0 for redback's forward models, peak_time for the sncosmo wrapper: both
    # are the model's time origin in MJD, so they are summarised the same way.
    time_key = time_parameter(cfg.model)
    t0_posterior = posterior[time_key].to_numpy() if time_key in posterior else None

    return {
        "object": cfg.object_name,
        "model": cfg.model,
        "label": cfg.label,
        "M_Ni": nickel_mass(posterior),
        "parameters": parameters,
        "log_evidence": float(result.log_evidence),
        "log_evidence_err": float(result.log_evidence_err),
        "n_samples": int(len(posterior)),
        "t0_mjd": float(np.median(t0_posterior)) if t0_posterior is not None else None,
        "t0_minus_first_detection_days": (
            float(np.median(t0_posterior) - transient.first_detection_mjd)
            if t0_posterior is not None
            else None
        ),
        "first_detection_mjd": float(transient.first_detection_mjd),
        "redshift_mode": cfg.redshift_mode,
        "extinction": extinction_summary(cfg),
        "upper_limit_sigma": float(transient.upper_limit_sigma),
        # What --likelihood resolved to. On the default it is redback's own
        # choice in `_fit_optical_transient`, reproduced here: the upper-limit
        # class when limits survived the band filter, a plain Gaussian when
        # none did.
        "likelihood": (
            LIKELIHOODS[cfg.likelihood]["cls"]
            if cfg.likelihood != DEFAULT_LIKELIHOOD
            else (
                "GaussianLikelihoodWithUpperLimits"
                if transient.has_upper_limits and detections is not None
                else "GaussianLikelihood"
            )
        ),
        "sigma_sys": (
            float(np.median(result.posterior["sigma"].to_numpy()))
            if "sigma" in result.posterior
            else None
        ),
        "n_detections": n_detections,
        "n_upper_limits": n_total - n_detections,
        "constraints": sorted(
            key
            for key, prior in priors.items()
            if isinstance(prior, bilby.core.prior.Constraint)
        ),
        "sampler": cfg.sampler,
        "nlive": cfg.nlive,
        "seed": cfg.seed,
        "datafile": str(cfg.datafile),
        "redback_version": redback.__version__,
        "bilby_version": bilby.__version__,
    }


def save_manifest(cfg: RunConfig, outdir: Path) -> Path:
    """Everything needed to reproduce this run, including the exact command."""
    import bilby

    manifest = {
        "config": {key: str(value) for key, value in asdict(cfg).items()},
        "label": cfg.label,
        "command": " ".join(sys.argv),
        "argv": sys.argv,
        "model_description": describe(cfg.model),
        "redback_version": redback.__version__,
        "bilby_version": bilby.__version__,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    path = outdir / f"{cfg.label}_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str))
    return path


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    cfg = resolve_config(args)

    logger.info(
        "redback %s | object %s | model %s",
        redback.__version__,
        cfg.object_name,
        cfg.model,
    )
    logger.info("%s", describe(cfg.model))

    np.random.seed(cfg.seed)
    import bilby

    bilby.core.utils.random.seed(cfg.seed)

    rows = read_photometry(cfg.datafile)
    if cfg.detections_only:
        kept = [row for row in rows if row["detected"]]
        logger.warning(
            "--detections-only: %d upper limits dropped, %d detections kept. "
            "They bound t0 from above, so its posterior widens, and the ln Z "
            "is not comparable with a run that keeps them.",
            len(rows) - len(kept),
            len(kept),
        )
        rows = kept
    reference_mjd = first_epoch_mjd(rows)

    transient = build_transient(
        rows,
        cfg.object_name,
        upper_limit_sigma=cfg.upper_limit_sigma,
        data_tmin=cfg.data_tmin,
        data_tmax=cfg.data_tmax,
    )

    priors = build_priors(cfg, reference_mjd, bands=transient.filtered_bands)
    kwargs_for_model = model_kwargs(
        cfg.model, transient.filtered_bands, sncosmo_model=cfg.sncosmo_model
    )

    outdir = cfg.outdir / cfg.model / cfg.object_name
    outdir.mkdir(parents=True, exist_ok=True)
    priors.to_file(outdir=str(outdir), label=f"{cfg.label}_effective")

    result = redback.fit_model(
        transient=transient,
        model=observed_model(cfg.model),
        prior=priors,
        # `likelihood` is not in fit_model's own signature but is a parameter of
        # the _fit_optical_transient it dispatches to, reached through **kwargs.
        # None is that function's default, so --likelihood upper-limits leaves
        # the choice entirely to redback.
        likelihood=build_likelihood(cfg, transient, kwargs_for_model),
        model_kwargs=kwargs_for_model,
        sampler=cfg.sampler,
        nlive=cfg.nlive,
        outdir=str(outdir),
        label=cfg.label,
        plot=False,
        resume=False,
        clean=True,
        seed=cfg.seed,
        # `sample` is a dynesty sampling scheme; pymultinest has no such
        # keyword and bilby would drop it with a warning, silently losing the
        # setting. MultiNest's own efficiency knobs are left at their defaults.
        **({"sample": "rslice"} if cfg.sampler == "dynesty" else {}),
    )

    figures = {
        "lightcurve": plot_redback.lightcurve(result, cfg, transient, outdir),
        "corner": plot_redback.corner(result, cfg, outdir),
    }
    plot_redback.write_posterior_samples(result, cfg.label, outdir)

    summary = summarise(result, cfg, transient, priors)
    summary["figures"] = {k: str(v) for k, v in figures.items() if v is not None}
    summary_path = outdir / f"{cfg.label}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    manifest_path = save_manifest(cfg, outdir)

    # Read the bounds off the prior that was actually sampled, not off cfg:
    # the sncosmo path bounds peak_time on the brightest detection, not on
    # --t0-min/--t0-max, so comparing against cfg silently tested nothing and
    # the warning never fired where it was needed most.
    time_key = time_parameter(cfg.model)
    drawn_t0 = result.posterior.get(time_key)
    prior = priors.get(time_key)
    if drawn_t0 is not None and prior is not None:
        edge_min = float(getattr(prior, "minimum", float("nan")))
        edge_max = float(getattr(prior, "maximum", float("nan")))
        span = edge_max - edge_min
        low, high = np.percentile(drawn_t0.to_numpy(), [5, 95])
        for bound, value, edge in (
            (f"the lower bound on {time_key}", low, edge_min),
            (f"the upper bound on {time_key}", high, edge_max),
        ):
            if span > 0 and abs(value - edge) < 0.02 * span:
                logger.warning(
                    "%s piles up against %s (%.5f MJD): the prior, not the "
                    "data, is constraining it. Widen the window and refit.",
                    time_key,
                    bound,
                    edge,
                )

    mni = summary["M_Ni"]
    if "median" in mni:
        logger.info(
            "M_Ni = %.4f (+%.4f -%.4f) Msun%s",
            mni["median"],
            mni["plus"],
            mni["minus"],
            f"   [kappa fixed at {cfg.kappa:g}]" if cfg.kappa is not None else "",
        )
    logger.info(
        "ln Z = %.2f +/- %.2f", summary["log_evidence"], summary["log_evidence_err"]
    )
    logger.info(
        "%s - first detection = %+.2f d",
        time_parameter(cfg.model),
        summary["t0_minus_first_detection_days"],
    )
    logger.info("summary:  %s", summary_path)
    logger.info("manifest: %s", manifest_path)
    for kind, path in summary["figures"].items():
        logger.info("%-10s: %s", kind, path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
