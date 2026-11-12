"""Light-curve and corner figures for a redback fit.

The layout is the Astro-COLIBRI one: the constants, the per-band panel with its
residual strip, the secondary MJD axis, the header and the vertical captions are
those of ``nmma_api/plot_bestfit.py``, so a redback fit and an NMMA fit of the
same target produce figures that read the same.

Only the palette is switchable, through ``--plot-theme``. ``print``, the
default, is a plain figure for a paper. ``astrocolibri`` is the branded dark
figure the app shows, with the logo, the slate ground and the filter colours the
frontend uses, and matches the NMMA figure.

One deliberate difference from the NMMA figure, because redback exports
different things:

The panel title carries the band name alone. NMMA appends a reduced
chi-square; that number conflates how good the model is with how precise the
photometry is, and on data at 0.04 mag it reads as a failure for a fit that is
accurate to 0.13 mag. The residual strip under each panel already shows the
same information per epoch, in units of sigma and with its sign, which is what
makes a systematic trend legible rather than a single scalar.

The shaded band is the 16-84 credible interval of the posterior draws, not a
model curve widened by a fixed error budget: the draws are the uncertainty
redback actually produces. A budget exists here only if ``--likelihood`` asked
for one of the two classes that sample it, and then it appears as the wider,
faint error bar behind each point, with the residual strip divided by that
same sigma. NMMA's ``--em-error-budget`` is the fixed-value
counterpart.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from ._models import DEFAULT_LIKELIHOOD, LIKELIHOODS, supports_extinction

if TYPE_CHECKING:  # pragma: no cover
    from ._fit import RunConfig

__all__ = ["lightcurve", "corner", "write_posterior_samples"]

# Filters in the convention a journal figure uses: the colour tracks the
# effective wavelength, blue to red, so a reader needs no legend to know which
# panel is which.
PRINT_BANDS = {
    "atlas c": "#00868b",
    "atlasc": "#00868b",
    "atlas o": "#e8711a",
    "atlaso": "#e8711a",
    "asas-sn g": "#2e8b57",
    "asas-sn v": "#6a5acd",
    "ztf g": "#1b7837",
    "ztfg": "#1b7837",
    "g": "#1b7837",
    "ztf r": "#c0392b",
    "ztfr": "#c0392b",
    "r": "#c0392b",
    "ztf i": "#6d2c1f",
    "ztfi": "#6d2c1f",
    "i": "#6d2c1f",
}


def _local_themes() -> dict:
    """Palettes from an optional sibling module, absent from this repository.

    The branded Astro-COLIBRI figure (slate ground, the frontend's filter
    colours, the logo) lives in ``_plot_astrocolibri.py``, which is not
    versioned here: it carries another project's identity, so it is kept out
    of the repository and out of a paper figure. When the file is present its
    palettes join :data:`THEMES` and ``--plot-theme`` offers them; when it is
    not, nothing changes and only ``print`` exists.
    """
    try:
        from ._plot_astrocolibri import THEMES as extra
    except ImportError:
        return {}
    return dict(extra)


# The looks the figure can take. Same layout and same panels in every case;
# only the palette changes.
THEMES = {
    "print": dict(
        background="white",
        foreground="black",
        title="black",
        model="#c2185b",
        grid="0.75",
        bands=PRINT_BANDS,
        fallback=(0.35, 0.35, 0.35),
        logo=None,
        url=None,
    ),
    **_local_themes(),
}

DEFAULT_THEME = "print"

NICE_BAND_NAMES = {
    "ztfg": "ZTF g-band",
    "ztfr": "ZTF r-band",
    "ztfi": "ZTF i-band",
    "ztf_g": "ZTF g-band",
    "ztf_r": "ZTF r-band",
    "ztf_i": "ZTF i-band",
    "atlasc": "ATLAS c-band",
    "atlaso": "ATLAS o-band",
    "atlas_c": "ATLAS c-band",
    "atlas_o": "ATLAS o-band",
    "lsstu": "LSST u-band",
    "lsstg": "LSST g-band",
    "lsstr": "LSST r-band",
    "lssti": "LSST i-band",
    "lsstz": "LSST z-band",
    "lssty": "LSST y-band",
    "asas-sn_g": "ASAS-SN g-band",
    "asas-sn_v": "ASAS-SN V-band",
    "aavso_vis.": "AAVSO visual",
    "aavso_vis": "AAVSO visual",
    "aavso_v": "AAVSO V-band",
    "aavso_b": "AAVSO B-band",
    "aavso_r": "AAVSO R-band",
    "aavso_i": "AAVSO I-band",
    "last_clear": "LAST clear",
}

# Layout, in inches. Edit here to resize the figure globally.
LAYOUT = dict(
    HDR_H=0.90,
    MAIN_H=2.20,
    RES_H=0.70,
    H_SEP=1.2,
    PNL_W=4.20,
    W_SEP=0.95,
    L_MAR=0.90,
    B_MAR=0.60,
    T_MAR=0.08,
    LOGO_W=1.10,
    LOGO_H=1.10,
    FONT_COL_W=0.14,
    GAP=0.05,
)

# Posterior draws behind the model band. Enough for a stable 16-84 interval
# without the evaluation dominating the runtime of a short fit.
N_DRAWS = 200


def theme_for(name: str | None) -> dict:
    """One of :data:`THEMES`, falling back to the default on an unknown name."""
    return THEMES.get(name or DEFAULT_THEME, THEMES[DEFAULT_THEME])


def colour_for_band(band: str, theme: dict):
    """The figure colour for a filter name, in this theme's palette."""
    return theme["bands"].get(band.strip().lower(), theme["fallback"])


def nice_band_name(band: str) -> str:
    """A human-readable name for a filter code, or the code itself."""
    if not band:
        return ""
    return NICE_BAND_NAMES.get(band.strip().lower(), band)


def _style(figure, ax, theme: dict, title=None) -> None:
    """Apply one theme to one axis."""
    ax.set_facecolor(theme["background"])
    figure.patch.set_facecolor(theme["background"])
    for side in ("bottom", "top", "right", "left"):
        ax.spines[side].set_color(theme["foreground"])
    ax.tick_params(which="both", axis="both", colors=theme["foreground"])
    ax.yaxis.label.set_color(theme["foreground"])
    ax.xaxis.label.set_color(theme["foreground"])
    if title is not None:
        ax.set_title(title, color=theme["title"], pad=15)


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


def effective_errors(cfg, result, errors, draws):
    """The sigma the chosen likelihood actually used, per point.

    With ``--likelihood upper-limits`` this is the photometric error and the
    array comes back untouched. The two budget likelihoods sample a ``sigma``
    on top of it, so the residual strip has to widen by the same amount, or the
    figure would show a bad fit that the inference never saw.

    The formulae are redback's own, as :data:`_models.LIKELIHOODS`
    records them: ``quadrature`` adds a constant in magnitudes, ``systematic``
    scales it by the model value.
    """
    name = getattr(cfg, "likelihood", DEFAULT_LIKELIHOOD)
    if name == DEFAULT_LIKELIHOOD or "sigma" not in result.posterior:
        return errors
    sigma = float(np.median(result.posterior["sigma"].to_numpy()))
    if LIKELIHOODS[name]["sigma"].startswith("sqrt(sigma_i**2 + model"):
        scale = np.median(draws, axis=0) if draws is not None else 0.0
        return np.sqrt(errors**2 + (scale * sigma) ** 2)
    return np.sqrt(errors**2 + sigma**2)


def _model_draws(result, model_name: str, times_mjd, bands, sncosmo_model=None):
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
    from ._models import (
        SNCOSMO_MODEL,
        model_kwargs,
        observed_model,
        sampled,
        sncosmo_keys,
        supports_extinction,
        time_parameter,
    )

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
        # sampled() can only report the template-independent keys for the
        # sncosmo path, so the source parameter (amplitude, or x0/x1/c) has to
        # come from sncosmo_keys. Filtering on sampled() alone dropped it and
        # the model call raised KeyError, leaving the figure with data but no
        # model curve, warned about and otherwise silent.
        if model_name == SNCOSMO_MODEL:
            wanted = set(sncosmo_keys(sncosmo_model))
        else:
            wanted = set(sampled(model_name)) | {time_parameter(model_name)}
        if supports_extinction(model_name):
            wanted |= {"av_mw", "av_host"}
        kwargs = model_kwargs(model_name, bands, sncosmo_model=sncosmo_model)
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

    from ._models import time_parameter

    time_key = time_parameter(cfg.model)
    t0_samples = (
        result.posterior[time_key].to_numpy() if time_key in result.posterior else None
    )

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

    theme = theme_for(getattr(cfg, "plot_theme", None))
    layout = LAYOUT
    n_bands = len(unique_bands)
    ncol = n_bands if n_bands <= 3 else (2 if n_bands == 4 else 3)
    nrow = int(np.ceil(n_bands / ncol))

    row_height = layout["MAIN_H"] + layout["RES_H"]
    right_margin = layout["LOGO_W"] + layout["FONT_COL_W"] * 2 - 7 * layout["GAP"]
    fig_w = (
        layout["L_MAR"]
        + ncol * layout["PNL_W"]
        + (ncol - 1) * layout["W_SEP"]
        + right_margin
    )
    fig_h = (
        layout["T_MAR"]
        + layout["HDR_H"]
        + nrow * row_height
        + (nrow - 1) * layout["H_SEP"]
        + layout["B_MAR"]
    )

    figure = plt.figure(figsize=(fig_w, fig_h))
    figure.patch.set_facecolor(theme["background"])

    panels_height = nrow * row_height + (nrow - 1) * layout["H_SEP"]
    gs_outer = GridSpec(
        2,
        1,
        figure=figure,
        left=layout["L_MAR"] / fig_w,
        right=1.0 - right_margin / fig_w,
        bottom=layout["B_MAR"] / fig_h,
        top=1.0 - layout["T_MAR"] / fig_h,
        hspace=0.0,
        height_ratios=[layout["HDR_H"], panels_height],
    )

    header = figure.add_subplot(gs_outer[0])
    header.set_facecolor(theme["background"])
    header.set_navigate(False)
    for spine in header.spines.values():
        spine.set_visible(False)
    header.set_xticks([])
    header.set_yticks([])

    logo_w = layout["LOGO_W"] / fig_w
    logo_h = layout["LOGO_H"] / fig_h
    logo_x = (fig_w - layout["GAP"] - layout["LOGO_W"]) / fig_w
    logo_y = (fig_h - layout["T_MAR"] - layout["LOGO_H"] - layout["GAP"]) / fig_h
    logo = theme.get("logo")
    if logo and Path(logo).is_file():
        ax_logo = figure.add_axes([logo_x, logo_y, logo_w, logo_h])
        ax_logo.imshow(plt.imread(logo))
        ax_logo.axis("off")
    if theme.get("url"):
        figure.text(
            logo_x + logo_w - layout["GAP"] / 3,
            logo_y + logo_h,
            theme["url"],
            ha="right",
            va="top",
            fontsize=8,
            color=theme["foreground"],
            transform=figure.transFigure,
        )

    header.text(
        0.5,
        logo_y + logo_h - 0.08,
        cfg.object_name.replace("_", " "),
        ha="center",
        va="center",
        fontsize=13,
        color=theme["title"],
        fontweight="bold",
        transform=header.transAxes,
    )

    # Vertical captions down the right margin, as on the NMMA figure.
    created_x = logo_y + logo_h
    figure.text(
        created_x,
        0.5,
        datetime.now(timezone.utc).strftime("created on %Y-%m-%d %H:%M UTC"),
        ha="center",
        va="center",
        fontsize=8,
        color=theme["foreground"],
        rotation=90,
        transform=figure.transFigure,
    )

    caption = [f"Model: {cfg.model}"]
    if cfg.model == "sncosmo_models":
        caption.append(cfg.sncosmo_model)
    if cfg.kappa is not None:
        caption.append(rf"$\kappa = {cfg.kappa:g}$")
    if cfg.ebv_mw and supports_extinction(cfg.model):
        caption.append(rf"$A_V^{{\rm MW}} \leq {3.1 * cfg.ebv_mw:.3f}$")
    figure.text(
        created_x - layout["GAP"] / fig_w - layout["FONT_COL_W"] / fig_w,
        0.5,
        "   ".join(caption),
        ha="center",
        va="center",
        fontsize=8,
        color=theme["foreground"],
        rotation=90,
        transform=figure.transFigure,
    )

    gs_panels = GridSpecFromSubplotSpec(
        nrow,
        ncol,
        subplot_spec=gs_outer[1],
        wspace=layout["W_SEP"] / layout["PNL_W"],
        hspace=layout["H_SEP"] / row_height,
    )

    # Blank any unused cell (three bands in a 2x2 layout, say).
    for index in range(n_bands, nrow * ncol):
        row, column = divmod(index, ncol)
        empty = figure.add_subplot(gs_panels[row, column])
        empty.set_facecolor(theme["background"])
        empty.axis("off")

    # Start the model grid at the earliest sampled explosion time, so the rise
    # implied by the posterior is visible rather than cropped at the first
    # datum. t0_base_model returns 1000 mag before t0, which the y limits clip.
    start_mjd = float(np.min(t0_samples)) if t0_samples is not None else times_mjd.min()
    model_times_mjd = np.linspace(start_mjd, times_mjd.max(), 200)
    model_phase = model_times_mjd - reference

    for index, band in enumerate(unique_bands):
        row, column = divmod(index, ncol)
        gs_cell = GridSpecFromSubplotSpec(
            2,
            1,
            subplot_spec=gs_panels[row, column],
            hspace=0.15,
            height_ratios=[layout["MAIN_H"], layout["RES_H"]],
        )
        main = figure.add_subplot(gs_cell[0])
        residual = figure.add_subplot(gs_cell[1], sharex=main)
        _style(figure, main, theme)
        _style(figure, residual, theme)
        plt.setp(main.get_xticklabels(), visible=False)

        # Secondary MJD axis on top, so a reader can go back to the alert.
        mjd_axis = main.secondary_xaxis(
            "top",
            functions=(
                lambda days, ref=reference: days + ref,
                lambda mjd, ref=reference: mjd - ref,
            ),
        )
        mjd_axis.set_xlabel("MJD", color=theme["foreground"], labelpad=4, fontsize=8)
        mjd_axis.tick_params(axis="x", colors=theme["foreground"], labelsize=7)
        mjd_axis.spines["top"].set_color(theme["foreground"])

        model_colour = theme["model"]
        band_colour = colour_for_band(band, theme)
        in_band = bands == band

        draws = _model_draws(
            result,
            cfg.model,
            model_times_mjd,
            [band] * model_times_mjd.size,
            getattr(cfg, "sncosmo_model", None),
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
                model_phase,
                low,
                high,
                facecolor=model_colour,
                alpha=0.2,
                label="68% credible",
            )

        if t0_samples is not None:
            main.axvline(
                float(np.median(t0_samples)) - reference,
                linestyle=":",
                color=theme["grid"],
                linewidth=1.2,
                label=rf"${time_key}$ (median)".replace("_", r"\_"),
            )

        seen = in_band & detected
        at_data = _model_draws(
            result,
            cfg.model,
            times_mjd[seen],
            [band] * int(seen.sum()),
            getattr(cfg, "sncosmo_model", None),
        )
        sigma_eff = effective_errors(cfg, result, errors[seen], at_data)

        # When a budget likelihood ran, the wider bar behind the photometric
        # one is the sigma the fit actually used. Drawn only then, so the
        # default figure carries exactly the published error bars and nothing
        # added on top of them.
        if not np.allclose(sigma_eff, errors[seen]):
            main.errorbar(
                phase[seen],
                magnitudes[seen],
                sigma_eff,
                fmt="none",
                ecolor=band_colour,
                alpha=0.3,
                elinewidth=3.0,
                zorder=4,
            )
        main.errorbar(
            phase[seen],
            magnitudes[seen],
            errors[seen],
            fmt="o",
            color=band_colour,
            ms=4,
            elinewidth=1.2,
            zorder=5,
        )
        limits = in_band & ~detected
        if limits.any():
            main.scatter(
                phase[limits],
                magnitudes[limits],
                marker="v",
                color=band_colour,
                s=20,
                zorder=5,
            )

        if at_data is not None and seen.any():
            median_at_data = np.median(at_data, axis=0)
            residual.scatter(
                phase[seen],
                (magnitudes[seen] - median_at_data) / sigma_eff,
                color=band_colour,
                s=20,
            )
        residual.axhline(0, linestyle="--", color=theme["grid"], linewidth=1)

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

        _style(figure, main, theme, title=nice_band_name(band))
        _style(figure, residual, theme)
        main.set_ylabel("AB magnitude", fontsize=8)
        legend = main.legend(
            fontsize=7,
            loc="best",
            facecolor=theme["background"],
            edgecolor=theme["foreground"],
        )
        for text in legend.get_texts():
            text.set_color(theme["foreground"])
        residual.set_ylabel(r"$\Delta\,(\sigma)$", fontsize=7)
        if row == nrow - 1:
            residual.set_xlabel(
                "Time since first detection [days]",
                fontsize=8,
                color=theme["foreground"],
            )
        else:
            plt.setp(residual.get_xticklabels(), visible=False)

    path = outdir / f"{cfg.label}_lightcurve.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
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
        # Exactly constant, which is what a DeltaFunction produces. NOT
        # np.allclose: its atol of 1e-8 calls any small-magnitude parameter
        # constant, and an sncosmo amplitude of 3e-13 varying by 1.3e-14 was
        # dropped as frozen, which left the corner plot with nothing to draw.
        if values.size and np.ptp(values) == 0.0:
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
