"""Light-curve and corner figures for a redback fit.

Follows the layout of ``plot_compare_models_public.py`` on the NMMA side: one
column per band, a 3:1 main/residual split, tab10 for models and Dark2 for
bands, dashed model curves over a shaded band, data as black-edged circles and
non-detections as triangles. Plain matplotlib, no branding, so the two
pipelines produce figures that can sit on the same page.

One deliberate difference. The NMMA figure annotates a reduced chi-square in
the legend, computed from the best-fit row it exports. Redback exports a
posterior and an evidence, not a maximum-likelihood row, so the same number
would have to be recomputed here from posterior draws and would not be the
NMMA quantity despite carrying the same symbol. The residual panel shows the
same information without that ambiguity.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from redback_models import supports_extinction

if TYPE_CHECKING:  # pragma: no cover
    from fit_redback import RunConfig

__all__ = ["lightcurve", "corner", "write_posterior_samples"]

# Model curves, as in the NMMA public script.
MODEL_COLORS = [cm.tab10(i) for i in range(10)]

# Photometry points, one colour per band.
BAND_COLORS = [cm.Dark2(i) for i in range(8)]

# Posterior draws behind the model band. Enough for a stable 16-84 interval
# without the evaluation dominating the runtime of a short fit.
N_DRAWS = 200


def write_posterior_samples(result, label: str, outdir: Path) -> Path:
    """Write the posterior as the ASCII table bilby produces elsewhere.

    redback keeps its posterior inside ``result.json`` only; exporting it in
    the common format lets the NMMA-side tooling in this repository read a
    redback run without special-casing it.

    Returns
    -------
    pathlib.Path
        Path of the written table.
    """
    path = outdir / f"{label}_posterior_samples.dat"
    path.parent.mkdir(parents=True, exist_ok=True)
    result.posterior.to_csv(path, sep=" ", index=False)
    return path


def _model_draws(result, model_name: str, times_mjd, bands):
    """Model magnitudes for ``N_DRAWS`` posterior samples.

    Evaluates the same callable the fit used, ``observed_model(model_name)``,
    i.e. redback's ``t0_base_model`` with the NaN guard, so the curve drawn
    is the curve that was fitted, sampled ``t0`` included.

    Parameters
    ----------
    result : redback.result.RedbackResult
        A completed fit.
    model_name : str
        Base model key; passed to ``t0_base_model`` as ``base_model``.
    times_mjd : array_like
        Epochs in **MJD, sorted ascending**. ``t0_base_model`` returns
        ``concatenate((pre_t0, post_t0))``, so unsorted input would be
        silently misaligned.
    bands : sequence of str
        One band per epoch, same length as ``times_mjd``.

    Returns
    -------
    numpy.ndarray or None
        Shape ``(n_draws, len(times_mjd))``, or None when the model cannot be
        evaluated. The caller then plots data only rather than failing.
    """
    from redback.utils import bands_to_frequency
    from redback_models import observed_model, sampled, supports_extinction

    times_mjd = np.asarray(times_mjd, dtype=float)
    if times_mjd.size and np.any(np.diff(times_mjd) < 0):
        logging.warning("model draws skipped: times are not sorted")
        return None

    try:
        posterior = result.posterior
        draws = posterior.sample(min(N_DRAWS, len(posterior)), replace=False)
        function = observed_model(model_name)
        # The model's own parameters, the time shift, and the extinction keys
        # its wrapper takes. The posterior also carries log_likelihood,
        # log_prior and the Constraint by-products, which are not model
        # arguments and would raise.
        wanted = set(sampled(model_name)) | {"t0"}
        if supports_extinction(model_name):
            wanted |= {"av_mw", "av_host"}
        kwargs = {
            "output_format": "magnitude",
            "base_model": model_name,
            "bands": bands,
            "frequency": bands_to_frequency(list(bands)),
        }
        curves = [
            np.asarray(
                function(
                    times_mjd,
                    **{key: row[key] for key in row.index if key in wanted},
                    **kwargs,
                )
            )
            for _, row in draws.iterrows()
        ]
        return np.vstack(curves)
    except Exception as exc:
        logging.warning("model draws unavailable: %s", exc)
        return None


def lightcurve(result, cfg: "RunConfig", transient, outdir: Path) -> Path | None:
    """Per-band light curve with residuals.

    Parameters
    ----------
    result : redback.result.RedbackResult
        A completed fit.
    cfg : RunConfig
        The configuration the fit ran with; supplies the model and label.
    transient : redback.transient.OpticalTransient
        The fitted transient, carrying the photometry and
        ``first_detection_mjd``.
    outdir : pathlib.Path
        Directory to write into.

    Returns
    -------
    pathlib.Path or None
        The written PDF, or None when the figure could not be produced.
    """
    # `import redback` turns matplotlib's usetex on through its own style
    # sheet; mathtext renders the few symbols here without a LaTeX install.
    matplotlib.rcParams["text.usetex"] = False

    # The transient is on the MJD scale (use_phase_model), which is what the
    # model must be evaluated on; the panels show days since the first
    # detection, which is what a reader can interpret.
    times_mjd = np.asarray(transient.x, dtype=float)
    reference = float(transient.first_detection_mjd)
    phase = times_mjd - reference
    magnitudes = np.asarray(transient.y, dtype=float)
    errors = np.asarray(transient.y_err, dtype=float)
    bands = np.asarray(transient.bands)

    t0_samples = result.posterior["t0"].to_numpy() if "t0" in result.posterior else None

    # `detections` is None, not an array, when every row is a detection:
    # redback sets the mask only when at least one row is an upper limit.
    raw_detections = getattr(transient, "detections", None)
    detected = (
        np.ones(times_mjd.size, dtype=bool)
        if raw_detections is None
        else np.asarray(raw_detections, dtype=bool)
    )

    unique_bands = sorted(set(bands.tolist()))
    if not unique_bands:
        return None

    ncol = len(unique_bands) if len(unique_bands) <= 3 else 2
    nrow = int(np.ceil(len(unique_bands) / ncol))
    figure, axes = plt.subplots(
        nrow * 2,
        ncol,
        figsize=(5 * ncol, 4 * nrow),
        sharex="col",
        gridspec_kw={"height_ratios": [3, 1] * nrow},
    )
    axes = np.atleast_2d(axes)
    subtitle = [cfg.model]
    if cfg.kappa is not None:
        subtitle.append(rf"$\kappa = {cfg.kappa:g}$")
    if cfg.ebv_mw and supports_extinction(cfg.model):
        subtitle.append(rf"$A_V^{{\rm MW}} \leq {3.1 * cfg.ebv_mw:.3f}$")
    figure.suptitle(
        f"{cfg.object_name.replace('_', ' ')}: " + ", ".join(subtitle),
        fontsize=13,
    )

    model_colour = MODEL_COLORS[0]
    # Start the model grid at the earliest sampled explosion time, so the rise
    # implied by the posterior is visible rather than cropped at the first
    # datum. t0_base_model returns 1000 mag before t0, which the y limits clip.
    start_mjd = float(np.min(t0_samples)) if t0_samples is not None else times_mjd.min()
    model_times_mjd = np.linspace(start_mjd, times_mjd.max(), 200)
    model_phase = model_times_mjd - reference

    for index, band in enumerate(unique_bands):
        row, column = divmod(index, ncol)
        main = axes[2 * row, column]
        residual = axes[2 * row + 1, column]
        band_colour = BAND_COLORS[index % len(BAND_COLORS)]
        in_band = bands == band

        draws = _model_draws(
            result,
            cfg.model,
            model_times_mjd,
            [band] * model_times_mjd.size,
        )
        if draws is not None:
            low, median, high = np.percentile(draws, [16, 50, 84], axis=0)
            main.plot(
                model_phase,
                median,
                color=model_colour,
                linewidth=2,
                linestyle="--",
                label=cfg.model,
            )
            main.fill_between(
                model_phase, low, high, facecolor=model_colour, alpha=0.15
            )

        if t0_samples is not None:
            main.axvline(
                float(np.median(t0_samples)) - reference,
                linestyle=":",
                color="gray",
                linewidth=1.2,
                label=r"$t_0$ (median)",
            )

        seen = in_band & detected
        main.errorbar(
            phase[seen],
            magnitudes[seen],
            errors[seen],
            fmt="o",
            color=band_colour,
            ms=6,
            markeredgecolor="black",
            elinewidth=1.2,
            zorder=10,
        )
        limits = in_band & ~detected
        if limits.any():
            main.scatter(
                phase[limits],
                magnitudes[limits],
                marker="v",
                color=band_colour,
                s=45,
                edgecolors="black",
                zorder=10,
            )

        at_data = _model_draws(
            result, cfg.model, times_mjd[seen], [band] * int(seen.sum())
        )
        if at_data is not None and seen.any():
            median_at_data = np.median(at_data, axis=0)
            residual.scatter(
                phase[seen],
                (magnitudes[seen] - median_at_data) / errors[seen],
                color=model_colour,
                s=20,
            )
        residual.axhline(0, linestyle="--", color="gray", linewidth=1)

        # Set the limits from the photometry before inverting: t0_base_model
        # returns 1000 mag before t0, and autoscaling on that would compress
        # the real light curve into a flat line at the top of the panel.
        # Detections only: an upper limit carries no error (NaN), and a band
        # made of limits alone would make nanmax return NaN and set_ylim fail.
        in_panel = magnitudes[in_band]
        panel_errors = errors[in_band & detected]
        if in_panel.size:
            margin = 0.5 + (float(np.max(panel_errors)) if panel_errors.size else 0.0)
            main.set_ylim(
                float(np.nanmin(in_panel)) - margin, float(np.nanmax(in_panel)) + margin
            )
        main.invert_yaxis()
        main.set_title(band)
        main.set_ylabel("AB magnitude")
        main.legend(fontsize=8, loc="best")
        residual.set_ylabel(r"$\Delta\,(\sigma)$")
        # Only the bottom row carries the x label: with two rows of bands it
        # would otherwise land on the panel title underneath.
        if row == nrow - 1:
            residual.set_xlabel("Days since first detection")

    # Blank any unused cell (three bands in a 2x2 layout, say).
    for index in range(len(unique_bands), nrow * ncol):
        row, column = divmod(index, ncol)
        axes[2 * row, column].axis("off")
        axes[2 * row + 1, column].axis("off")

    path = outdir / f"{cfg.label}_lightcurve.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(path, dpi=150, bbox_inches="tight")
    figure.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


def physical_posterior_columns(posterior) -> list[str]:
    """The posterior columns worth cornering.

    Drops bilby's bookkeeping (``log_likelihood``, ``log_prior``), the
    ``Constraint`` by-products redback's conversion functions add
    (``e_rot_constraint``, ``t_nebula_min``, ...), and any column held
    constant by a DeltaFunction, corner cannot plot a zero-width
    distribution and raises on it.
    """
    columns = []
    for column in posterior.columns:
        if column in ("log_likelihood", "log_prior"):
            continue
        if column.startswith("log_"):
            continue
        if column.endswith(("_constraint", "_ratio")) or column == "t_nebula_min":
            continue
        values = posterior[column].to_numpy()
        if not np.issubdtype(values.dtype, np.number):
            continue
        if np.allclose(values, values[0]):
            continue
        columns.append(column)
    return columns


def corner(result, cfg: "RunConfig", outdir: Path) -> Path | None:
    """Corner plot of the sampled parameters.

    Returns
    -------
    pathlib.Path or None
        The written PDF, or None when the plot could not be produced.
    """
    matplotlib.rcParams["text.usetex"] = False
    path = outdir / f"{cfg.label}_corner.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        columns = physical_posterior_columns(result.posterior)
        if not columns:
            logging.warning("corner plot skipped: no varying parameter")
            return None
        result.plot_corner(parameters=columns, save=True, filename=str(path), dpi=150)
        plt.close("all")
        # NMMA's comparison tooling looks for PNG; keep both.
        png = path.with_suffix(".png")
        if path.exists() and not png.exists():
            result.plot_corner(
                parameters=columns, save=True, filename=str(png), dpi=150
            )
            plt.close("all")
        return path
    except Exception as exc:
        logging.warning("corner plot failed: %s", exc)
        return None
