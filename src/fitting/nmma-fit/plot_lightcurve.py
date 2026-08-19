"""Plot the best-fit light curve for an nmma fit_nmma.py run.

Same per-filter [data + best-fit model + residuals] panel layout, header,
right-margin provenance columns and secondary MJD axis as the reference
light-curve figure this was ported from, but with a plain matplotlib style
(default colors, white background, no logo/branding). Reads the
*_bestfit_params.json written by `lightcurve-analysis --bestfit`.

Usage:
    .venv-nmma/bin/python plot_lightcurve.py \\
        --object ZTF21aapjmda --label v19-1993j-corr_ZTF21aapjmda \\
        --data data/processed/ZTF21aapjmda.dat --outdir results/
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from fit_nmma import get_trigger_time_from_datafile
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from nmma.em.io import load_em_observations
from nmma.em.utils import get_filtered_mag

logger = logging.getLogger(__name__)

# Plain matplotlib colors, one per filter (in first-seen order) --
# replaces the branded per-band color map of the reference figure.
_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]

# Same numeric layout as the reference figure (inches), minus the logo
# column: R_MAR there was LOGO_W + FONT_COL_W*2 - 7*GAP; without a logo we
# just reserve room for the two vertical text columns.
_LAYOUT = dict(
    HDR_H=0.90,
    MAIN_H=2.20,
    RES_H=0.70,
    H_SEP=1.2,
    PNL_W=4.20,
    W_SEP=0.95,
    L_MAR=0.90,
    B_MAR=0.60,
    T_MAR=0.08,
    FONT_COL_W=0.14,
    GAP=0.05,
)

_NICE_BAND_NAMES = {
    "ztfg": "ZTF g-band",
    "ztfr": "ZTF r-band",
    "ztfi": "ZTF i-band",
    "atlasc": "ATLAS c-band",
    "atlaso": "ATLAS o-band",
    "lsstu": "LSST u-band",
    "lsstg": "LSST g-band",
    "lsstr": "LSST r-band",
    "lssti": "LSST i-band",
    "lsstz": "LSST z-band",
    "lssty": "LSST y-band",
}


def nice_band_name(band: str) -> str:
    """Human-readable filter name; falls back to the raw code if unknown."""
    return _NICE_BAND_NAMES.get(band.strip().lower(), band) if band else ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", required=True, dest="obj")
    parser.add_argument("--label", required=True)
    parser.add_argument("--data", required=True, dest="datafile", type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--error-budget", type=float, default=0.25, dest="error_budget")
    return parser.parse_args()


def plot_lightcurve(
    obj: str, label: str, model: str, datafile: Path, outdir: Path, error_budget: float
) -> Path:
    base = outdir / obj
    with open(base / f"{label}_bestfit_params.json") as fh:
        best = json.load(fh)

    data = load_em_observations(str(datafile), format="observations")
    trigger_time = get_trigger_time_from_datafile(datafile)

    obs_times = np.asarray(best["obs_times"])
    mag = {filt: np.asarray(vals) for filt, vals in best["Magnitudes"].items()}
    filters_plot = [f for f in mag if f in data]

    all_y = np.concatenate([data[f]["mag"] for f in filters_plot])
    all_y = all_y[np.isfinite(all_y)]
    y_rng = max(all_y.max() - all_y.min(), 0.1)
    ylim = (
        all_y.max() + error_budget + 0.4 * y_rng,
        all_y.min() - error_budget - 0.4 * y_rng,
    )
    xlim = (obs_times.min(), obs_times.max())

    # Figure layout
    L = _LAYOUT
    n = len(filters_plot)
    ncol = n if n <= 3 else (2 if n == 4 else 3)
    nrow = int(np.ceil(n / ncol))
    row_h = L["MAIN_H"] + L["RES_H"]
    r_mar = L["FONT_COL_W"] * 2 + L["GAP"] * 3

    fig_w = L["L_MAR"] + ncol * L["PNL_W"] + (ncol - 1) * L["W_SEP"] + r_mar
    fig_h = (
        L["T_MAR"] + L["HDR_H"] + nrow * row_h + (nrow - 1) * L["H_SEP"] + L["B_MAR"]
    )

    fig = plt.figure(figsize=(fig_w, fig_h))
    panels_h = nrow * row_h + (nrow - 1) * L["H_SEP"]
    gs_outer = GridSpec(
        2,
        1,
        figure=fig,
        left=L["L_MAR"] / fig_w,
        right=1.0 - r_mar / fig_w,
        bottom=L["B_MAR"] / fig_h,
        top=1.0 - L["T_MAR"] / fig_h,
        # The reference figure uses hspace=0.0 here and relies on the logo
        # box geometry (logo_y + logo_h - 0.08) to keep the header text
        # clear of the top panel row. Without a logo we have no such
        # reference point, so a small explicit gap does the same job.
        hspace=L["H_SEP"] / L["HDR_H"],
        height_ratios=[L["HDR_H"], panels_h],
    )

    # Header: candidate name, centered
    ax_hdr = fig.add_subplot(gs_outer[0])
    ax_hdr.axis("off")
    ax_hdr.text(
        0.5,
        1.0,
        (obj or "").replace("_", " "),
        ha="center",
        va="top",
        fontsize=13,
        fontweight="bold",
        transform=ax_hdr.transAxes,
    )

    # Right margin: "Model" / "created on" vertical text columns
    # (same provenance columns as the reference figure; no logo/URL column)
    fig.text(
        1.0 - (L["FONT_COL_W"] / 2) / fig_w,
        0.5,
        datetime.now(timezone.utc).strftime("created on %Y-%m-%d %H:%M UTC"),
        ha="center",
        va="center",
        fontsize=8,
        rotation=90,
        transform=fig.transFigure,
    )
    fig.text(
        1.0 - (L["FONT_COL_W"] * 1.5 + L["GAP"]) / fig_w,
        0.5,
        f"Model: {model}",
        ha="center",
        va="center",
        fontsize=8,
        rotation=90,
        transform=fig.transFigure,
    )

    gs_panels = GridSpecFromSubplotSpec(
        nrow,
        ncol,
        subplot_spec=gs_outer[1],
        wspace=L["W_SEP"] / L["PNL_W"],
        hspace=L["H_SEP"] / row_h,
    )

    # Per-filter panels
    for cnt, filt in enumerate(filters_plot):
        color = _COLORS[cnt % len(_COLORS)]
        r, c = divmod(cnt, ncol)
        gs_cell = GridSpecFromSubplotSpec(
            2,
            1,
            subplot_spec=gs_panels[r, c],
            hspace=0.15,
            height_ratios=[L["MAIN_H"], L["RES_H"]],
        )
        ax_sum = fig.add_subplot(gs_cell[0])
        ax_delta = fig.add_subplot(gs_cell[1], sharex=ax_sum)
        plt.setp(ax_sum.get_xticklabels(), visible=False)

        ax_sum.set_ylabel("AB magnitude", fontsize=8)
        ax_delta.set_ylabel(r"$\Delta\,(\sigma)$", fontsize=7)
        if r == nrow - 1:
            ax_delta.set_xlabel("Time since trigger [days]", fontsize=8)
        else:
            plt.setp(ax_delta.get_xticklabels(), visible=False)

        ax_mjd = ax_sum.secondary_xaxis(
            "top",
            functions=(lambda t: t + trigger_time, lambda mjd: mjd - trigger_time),
        )
        ax_mjd.set_xlabel("MJD", labelpad=4, fontsize=8)
        ax_mjd.tick_params(axis="x", labelsize=7)

        t_obs = data[filt]["time"] - trigger_time
        y_obs, sig_obs = data[filt]["mag"], data[filt]["mag_error"]
        valid = ~np.isnan(y_obs)
        t_obs, y_obs, sig_obs = t_obs[valid], y_obs[valid], sig_obs[valid]
        det = np.isfinite(sig_obs)

        ax_sum.errorbar(
            t_obs[det], y_obs[det], sig_obs[det], fmt="o", color=color, ms=4, zorder=5
        )
        if (~det).any():
            ax_sum.scatter(
                t_obs[~det], y_obs[~det], marker="v", color=color, s=20, zorder=5
            )

        mag_plot = get_filtered_mag(mag, filt)
        ax_sum.plot(obs_times, mag_plot, color="black", linewidth=2, linestyle="--")
        ax_sum.fill_between(
            obs_times,
            mag_plot + error_budget,
            mag_plot - error_budget,
            facecolor="black",
            alpha=0.15,
        )

        if det.any():
            mag_interp = np.interp(t_obs[det], obs_times, mag_plot)
            diff = mag_interp - y_obs[det]
            sigma_tot = np.sqrt(sig_obs[det] ** 2 + error_budget**2)
            ax_delta.scatter(t_obs[det], diff / sigma_tot, color=color, s=20)
            ax_delta.axhline(0, linestyle="--", color="gray", linewidth=1)
            chi2_str = f"{np.sum((diff / sigma_tot) ** 2) / det.sum():.2f}"
        else:
            chi2_str = "N/A"

        ax_sum.set_title(
            rf"{nice_band_name(filt)}:  $\chi^2_{{\mathrm{{red}}}} = {chi2_str}$",
            fontsize=9,
        )
        ax_sum.set_xlim(xlim)
        ax_sum.set_ylim(ylim)
        ax_delta.set_xlim(xlim)

    plot_path = base / f"{label}_lightcurve.png"
    fig.savefig(plot_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", plot_path)
    return plot_path


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    model = args.label.rsplit(f"_{args.obj}", 1)[0]
    plot_lightcurve(
        args.obj, args.label, model, args.datafile, args.outdir, args.error_budget
    )


if __name__ == "__main__":
    main()
