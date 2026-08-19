"""Fit a ZTF object with NMMA, baseline only (SALT3/Nugent templates).

Builds and runs the `lightcurve-analysis` command with nmma>=1.0.0 flags.
NMMA has no physical parameters (M_Ni, M_ej): it serves as a
comparison/sanity-check, not a primary measurement. Do not use
--em-model salt3 on peculiar objects (91bg, 02cx): outside its domain
of validity.

Extinction: Ebv is left as a free nuisance parameter (--use-Ebv), bounded
by --Ebv-max = raw SFD E(B-V) at (ra, dec) times the Schlafly & Finkbeiner
(2011) recalibration factor (0.86), an upper bound rather than a fixed
value, since host-galaxy reddening is unknown. SFD here, not the Planck GNILC map
used for the actual MW dereddening in redback/mosfit
(src/ingest/extinction.py): the SF2011 factor is specifically a correction
for SFD, and this is only a nuisance-parameter bound, not a measurement.
--trigger-time is required in nmma-1.0 and is derived from the first ISOT
timestamp in the .dat file.

--em-transient-class: only SN templates (SALT3, Nugent, v19-*) are
relevant here, so it defaults to "supernova".

Usage:
    .venv-nmma/bin/python fit_nmma.py \\
        --object ZTF20abaunmw --ra 149.87 --dec 2.21 \\
        --data data/processed/ZTF20abaunmw.dat --model nugent-sn1a \\
        --prior config/priors/nugent-sn1a.prior --outdir results/
"""

import argparse
import json
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import bilby
import matplotlib
from astropy.time import Time

matplotlib.use("Agg")

from nmma_priors import configure_prior

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ingest"))
from extinction import EBV_SF2011, get_sfd_ebv
from time_windows import em_window

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunConfig:
    obj: str
    ra: float  # degrees, ICRS
    dec: float  # degrees, ICRS
    datafile: Path
    model: str
    prior_file: Path
    outdir: Path
    data_tmin: float
    data_tmax: float
    em_tstep: float
    error_budget: float
    nlive: int
    sampler: str
    interpolation_type: str
    em_transient_class: str
    seed: int
    cpus: int | None
    z_known: float | None
    ebv_max: float | None
    local_plot: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--object",
        required=True,
        dest="obj",
        help="ZTF object name (e.g. ZTF20abaunmw)",
    )
    parser.add_argument(
        "--ra", type=float, required=True, help="Right ascension, degrees (ICRS)"
    )
    parser.add_argument(
        "--dec", type=float, required=True, help="Declination, degrees (ICRS)"
    )
    parser.add_argument(
        "--data",
        required=True,
        dest="datafile",
        type=Path,
        help="Photometry in NMMA .dat format",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="NMMA template (e.g. nugent-sn1a, v19-1993j-corr; not salt3 on peculiars)",
    )
    parser.add_argument(
        "--prior", required=True, dest="prior_file", type=Path, help="NMMA prior file"
    )
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--data-tmin", type=float, default=0.0, dest="data_tmin")
    parser.add_argument("--data-tmax", type=float, default=47.0, dest="data_tmax")
    parser.add_argument("--em-tstep", type=float, default=0.25, dest="em_tstep")
    parser.add_argument("--error-budget", type=float, default=0.25, dest="error_budget")
    parser.add_argument("--nlive", type=int, default=2048)
    # dynesty currently crashes when combined with --use-Ebv on this nmma
    # version (shape mismatch between search_parameter_keys and samples);
    # pymultinest is the validated default until that's fixed upstream.
    parser.add_argument("--sampler", default="pymultinest")
    parser.add_argument(
        "--interpolation-type",
        default="sklearn_gp",
        dest="interpolation_type",
        help="sklearn_gp (CPU); keras is upstream's default but requires a GPU",
    )
    parser.add_argument(
        "--em-transient-class",
        default="supernova",
        dest="em_transient_class",
        help="nmma-1.0 class (default: supernova, the only one relevant for SALT3/Nugent here)",
    )
    parser.add_argument("--seed", type=int, default=42, dest="seed")
    parser.add_argument("--cpus", type=int, default=None)
    parser.add_argument(
        "--z-known",
        type=float,
        default=None,
        dest="z_known",
        help="Known spectroscopic redshift (TNS); fixes luminosity_distance instead of leaving it free",
    )
    parser.add_argument(
        "--ebv-max",
        type=float,
        default=None,
        dest="ebv_max",
        help="Pre-computed Ebv-max (SFD x SF2011). If given, skips the dustmaps query "
        "(useful if the run environment can't reach the network); if omitted, "
        "computed here from --ra/--dec via extinction.get_sfd_ebv",
    )
    parser.add_argument(
        "--local-plot",
        action="store_true",
        dest="local_plot",
        help="Also render plot_lightcurve.py's plot ({label}_lightcurve.png), in addition "
        "to nmma's native --plot output ({label}_bestfit_lightcurves.png)",
    )
    args = parser.parse_args()
    return RunConfig(**vars(args))


def get_trigger_time_from_datafile(datafile: Path) -> float:
    """MJD of the first ISOT timestamp in the .dat file (--trigger-time, required in nmma-1.0)."""
    with open(datafile) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts:
                return Time(parts[0], format="isot").mjd
    raise ValueError(f"No valid ISOT timestamp in {datafile}")


def validate_environment(cfg: RunConfig) -> None:
    if not cfg.datafile.exists():
        raise FileNotFoundError(f"--data not found: {cfg.datafile}")
    if not cfg.prior_file.exists():
        raise FileNotFoundError(f"--prior not found: {cfg.prior_file}")
    if shutil.which("lightcurve-analysis") is None:
        raise RuntimeError(
            "'lightcurve-analysis' not found in PATH (.venv-nmma active?)"
        )


def timeshift_from_prior(prior_path: Path):
    """Read the timeshift prior back out as an em_window()-compatible value."""
    priors = bilby.gw.prior.PriorDict(str(prior_path))
    ts_prior = priors.get("timeshift")
    if ts_prior is None:
        return None
    minimum, maximum = (
        getattr(ts_prior, "minimum", None),
        getattr(ts_prior, "maximum", None),
    )
    if minimum is not None and maximum is not None and minimum != maximum:
        return {"min": minimum, "max": maximum}
    peak = getattr(ts_prior, "peak", None)
    return peak if peak is not None else None


def build_lightcurve_cmd(
    cfg: RunConfig,
    trigger_time: float,
    prior_path: Path,
    em_tmin: float,
    em_tmax: float,
    ebv_max: float,
) -> list[str]:
    label = f"{cfg.model}_{cfg.obj}"
    model_outdir = cfg.outdir / cfg.obj
    cmd = [
        "lightcurve-analysis",
        "--em-model",
        cfg.model,
        "--outdir",
        str(model_outdir),
        "--light-curve-data",
        str(cfg.datafile),
        "--prior-file",
        str(prior_path),
        "--label",
        label,
        "--em-tmin",
        str(em_tmin),
        "--em-tmax",
        str(em_tmax),
        "--data-tmin",
        str(cfg.data_tmin),
        "--data-tmax",
        str(cfg.data_tmax),
        "--em-tstep",
        str(cfg.em_tstep),
        "--em-error-budget",
        str(cfg.error_budget),
        "--nlive",
        str(cfg.nlive),
        "--interpolation-type",
        cfg.interpolation_type,
        "--generation-seed",
        str(cfg.seed),
        "--use-Ebv",
        "--Ebv-max",
        str(ebv_max),
        "--bestfit",
        # --bestfit alone only writes the JSON (params + magnitudes); the
        # native lightcurve PNG needs --plot separately
        # (nmma/em/lightcurve_handling.py). --xlim frames the plot on the
        # data window (+/-1 day), not the full em window: with a wide
        # timeshift prior (e.g. +/-30 d) the em window is mostly empty
        # margin around the actual data. nargs="*" with no type=float:
        # needs two separate tokens, not a single "min,max" string.
        "--plot",
        "--xlim",
        str(cfg.data_tmin - 1),
        str(cfg.data_tmax + 1),
        "--local-model-only",
        "--sampler",
        cfg.sampler,
        "--trigger-time",
        str(trigger_time),
    ]
    if cfg.em_transient_class:
        cmd += ["--em-transient-class", cfg.em_transient_class]
    if cfg.cpus:
        cmd += ["--cpus", str(cfg.cpus)]
    return cmd


def save_run_manifest(cfg: RunConfig, cmd: list[str], trigger_time: float) -> Path:
    model_outdir = cfg.outdir / cfg.obj
    model_outdir.mkdir(parents=True, exist_ok=True)
    manifest = {
        **{k: str(v) for k, v in cfg.__dict__.items()},
        "trigger_time_mjd": trigger_time,
        "cmd": cmd,
    }
    manifest_path = model_outdir / f"{cfg.model}_{cfg.obj}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


def main() -> None:
    cfg = parse_args()
    logging.basicConfig(level=logging.INFO)
    logger.info("seed: %s", cfg.seed)

    validate_environment(cfg)
    trigger_time = get_trigger_time_from_datafile(cfg.datafile)
    logger.info("trigger-time (MJD, first observation): %.5f", trigger_time)

    prior_path = configure_prior(
        cfg.prior_file,
        outdir=cfg.outdir / cfg.obj,
        label=f"{cfg.model}_{cfg.obj}_prior",
        z_known=cfg.z_known,
    )
    em_tmin, em_tmax = em_window(
        cfg.data_tmin, cfg.data_tmax, timeshift_from_prior(prior_path)
    )
    logger.info(
        "em window: [%.2f, %.2f] (data [%.2f, %.2f])",
        em_tmin,
        em_tmax,
        cfg.data_tmin,
        cfg.data_tmax,
    )

    if cfg.ebv_max is not None:
        ebv_max = cfg.ebv_max
        logger.info("Ebv-max: %.5f (pre-computed, --ebv-max)", ebv_max)
    else:
        ebv_max = get_sfd_ebv(cfg.ra, cfg.dec) * EBV_SF2011
        logger.info("Ebv-max: %.5f (SFD x SF2011, live query)", ebv_max)

    cmd = build_lightcurve_cmd(cfg, trigger_time, prior_path, em_tmin, em_tmax, ebv_max)
    save_run_manifest(cfg, cmd, trigger_time)

    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(
            "lightcurve-analysis failed (exit %d):\n%s",
            result.returncode,
            result.stderr[-2000:],
        )
        sys.exit(1)

    if cfg.local_plot:
        from plot_lightcurve import plot_lightcurve

        plot_lightcurve(
            cfg.obj,
            f"{cfg.model}_{cfg.obj}",
            cfg.model,
            cfg.datafile,
            cfg.outdir,
            cfg.error_budget,
        )

    logger.info("Done. Results in %s", cfg.outdir / cfg.obj)


if __name__ == "__main__":
    main()
