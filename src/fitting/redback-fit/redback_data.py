"""Photometry loading and transient construction for the redback fits.

Reads the ``<isot> <band> <mag> <err>`` files in ``data/processed/``, the same
format the NMMA side consumes, so both fitters see byte-identical photometry
and a disagreement between them is a disagreement about models.

Every transient is built as a ``redback.transient.OpticalTransient`` in
magnitude mode, whatever the object's class. ``redback.fit_model`` routes it to
``_fit_optical_transient``, which selects ``GaussianLikelihoodWithUpperLimits``
when non-detections are present.

Not because the specialised classes would behave differently, ``Supernova``
and ``TDE`` both subclass ``OpticalTransient`` and are dispatched by the very
same ``isinstance`` branch, so they get the same likelihood. The reason is that
the five catalogued models span the supernova, TDE and shock-powered families,
and the subclasses add only survey-specific loaders
(``from_open_access_catalogue`` and friends) that this pipeline does not use.
One generic class keeps one code path where three would otherwise be needed,
and keeps the data identical across models, which is what makes their ln Z
comparable.

Two invariants this module enforces, both required by
``redback_models.observed_model``:

- **times are sorted ascending.** ``t0_base_model`` returns
  ``concatenate((pre_t0, post_t0))``, so an unsorted time array is silently
  misaligned with the data rather than raising.
- **the transient carries MJD.** It is built with ``use_phase_model=True`` and
  ``time_mjd=``, so ``Transient.x`` is MJD and the sampled ``t0`` is on the
  same scale.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# The one sentinel this format defines, matching the NMMA side: an infinite
# magnitude error marks an upper limit. Anything else non-finite (a NaN error,
# say) is not a declared limit and carries no usable uncertainty either, so it
# is dropped with a warning rather than given an invented one, fabricating a
# 0.1 mag error, as this reader used to, puts a number the data never reported
# into the likelihood.
UPPER_LIMIT_ERROR_IS_INF = True


def read_photometry(datafile: Path) -> list[dict]:
    """Parse one ``.dat`` photometry file.

    Parameters
    ----------
    datafile : pathlib.Path
        File of ``<isot> <band> <mag> <err>`` rows. ``err`` of ``inf`` marks
        an upper limit, the convention the ingest step writes.

    Returns
    -------
    list of dict
        One entry per usable row with keys ``mjd``, ``band``, ``mag``,
        ``mag_err`` and ``detected``, **sorted by time**, an invariant
        ``build_transient`` re-checks and ``t0_base_model`` depends on.

    Raises
    ------
    ValueError
        If the file holds no usable row.
    """
    from astropy.time import Time

    rows: list[dict] = []
    with open(datafile) as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            isot, band, mag, err = parts[:4]
            try:
                mjd = float(Time(isot, format="isot").mjd)
                mag_value = float(mag)
                err_value = float(err)
            except Exception:
                logger.debug("skipping unparsable row: %s", line)
                continue

            # +inf is the upper-limit sentinel. Any other non-finite error
            # (NaN, -inf) is neither a limit nor a measurement: skip the row
            # loudly instead of inventing an uncertainty for it.
            detected = not math.isinf(err_value)
            if detected and not math.isfinite(err_value):
                logger.warning(
                    "%s: dropping row with a non-finite, non-infinite error "
                    "(no usable uncertainty, and not a declared limit): %s",
                    datafile.name,
                    line,
                )
                continue
            if math.isinf(err_value) and err_value < 0:
                logger.warning(
                    "%s: dropping row with a negative infinite error: %s",
                    datafile.name,
                    line,
                )
                continue
            rows.append(
                {
                    "mjd": mjd,
                    "band": band,
                    "mag": mag_value,
                    "mag_err": err_value if detected else float("nan"),
                    "detected": detected,
                }
            )

    if not rows:
        raise ValueError(f"No usable photometry in {datafile}")
    rows.sort(key=lambda row: row["mjd"])
    return rows


def first_detection_mjd(rows: list[dict]) -> float:
    """MJD of the earliest detection.

    The reference epoch for every time window and for the ``t0`` prior: unlike
    a derived explosion time it is a datum, not an inference, so a prior
    centred on it does not smuggle in an assumption about the rise.

    Raises
    ------
    ValueError
        If no row is a detection.
    """
    for row in rows:
        if row["detected"]:
            return float(row["mjd"])
    raise ValueError("No detection in the photometry; cannot anchor t0")


def build_transient(
    rows: list[dict],
    name: str,
    *,
    upper_limit_sigma: float = 5.0,
    data_tmin: float | None = None,
    data_tmax: float | None = None,
):
    """Assemble a redback transient in magnitude mode, on the MJD scale.

    Parameters
    ----------
    rows : list of dict
        Photometry from :func:`read_photometry`. Magnitudes are used as
        observed: extinction belongs in the forward model
        (``redback_priors`` samples av_mw), never in the data.
    name : str
        Object identifier, used as the transient's name.
    upper_limit_sigma : float, optional
        The sigma level of the non-detections, which is redback's own meaning
        for this parameter. It sets the width of the survival term:
        ``sigma_UL = (2.5 / ln 10) / upper_limit_sigma``. The default of 5
        matches the ATLAS 5-sigma limiting magnitudes the ingest step writes
        (service paper, section 3); redback's own default of 3 would describe
        data this pipeline does not have. It matters: a model exceeding a limit
        by 1 mag is penalised -5.9 at sigma level 3 and -13.1 at 5.
    data_tmin, data_tmax : float, optional
        Window in days **relative to the first detection**. ``data_tmin`` may
        be negative to keep the pre-explosion limits, which are what constrain
        ``t0``. None keeps everything on that side.

    Returns
    -------
    redback.transient.OpticalTransient
        With ``first_detection_mjd`` attached, so the caller and the figures
        share one reference epoch.

    Raises
    ------
    ValueError
        If the rows are not sorted by time, if there is no detection, or if the
        window empties the data.
    """
    from redback.transient import OpticalTransient

    mjd = np.array([row["mjd"] for row in rows], dtype=float)
    if np.any(np.diff(mjd) < 0):
        raise ValueError(
            f"{name}: photometry is not sorted by time. t0_base_model "
            "concatenates pre- and post-t0 output, so unsorted input would be "
            "silently misaligned with the data."
        )

    detected = np.array([row["detected"] for row in rows], dtype=bool)
    if not detected.any():
        raise ValueError(f"{name}: no detection in the photometry")

    # Note: a limit with a non-finite magnitude is checked by redback itself,
    # in Transient.validate_upper_limits(), which warns that such points
    # "cannot be plotted or used in likelihood fitting". _fit_optical_transient
    # then falls back to a detections-only GaussianLikelihood, dropping every
    # limit. Nothing is re-checked here.

    reference = first_detection_mjd(rows)
    phase = mjd - reference

    keep = np.ones(len(rows), dtype=bool)
    if data_tmin is not None:
        keep &= phase >= data_tmin
    if data_tmax is not None:
        keep &= phase <= data_tmax
    if not keep.any():
        raise ValueError(f"{name}: no photometry left after the time cut")
    if not detected[keep].any():
        raise ValueError(f"{name}: the time cut removed every detection")
    if not keep.all():
        logger.info(
            "%s: keeping %d of %d rows in phase [%s, %s] d",
            name,
            int(keep.sum()),
            len(rows),
            data_tmin if data_tmin is not None else "-inf",
            data_tmax if data_tmax is not None else "+inf",
        )

    magnitudes = np.array([row["mag"] for row in rows], dtype=float)
    bands = np.array([row["band"] for row in rows])

    # The measured errors, passed through untouched, which is what redback's
    # own loader does (`magnitude_err = np.array(df["e_magnitude"])`, no
    # substitution). No systematic floor is folded in here: redback has no such
    # concept, and adding NMMA's would compress the relative weighting of this
    # data set by a factor of 14 (best detection 0.043 mag, worst 0.207, so a
    # 23x weight ratio that a 0.25 mag floor flattens to 1.6x).
    #
    # Upper limits keep their NaN. Their error is never read: the survival term
    # scales with upper_limit_sigma alone (measured: 0.25, inf and NaN give the
    # same lnL), and the figures draw limits as arrows with no error bar.
    errors = np.array([row["mag_err"] for row in rows], dtype=float)

    mjd, magnitudes, errors, detected, bands, phase = (
        mjd[keep],
        magnitudes[keep],
        errors[keep],
        detected[keep],
        bands[keep],
        phase[keep],
    )

    transient = OpticalTransient(
        name=name,
        data_mode="magnitude",
        # `time` is informative only (figures, phase axis); the likelihood
        # reads `time_mjd` because use_phase_model is set.
        time=phase,
        time_mjd=mjd,
        magnitude=magnitudes,
        magnitude_err=errors,
        bands=bands,
        use_phase_model=True,
        # Only pass the mask when it carries information: redback selects the
        # upper-limit-aware likelihood on its presence.
        detections=None if detected.all() else detected,
        upper_limit_sigma=upper_limit_sigma,
    )
    transient.first_detection_mjd = reference
    logger.info(
        "%s: %d points (%d detections, %d limits) in %d bands, "
        "first detection MJD %.5f",
        name,
        len(magnitudes),
        int(detected.sum()),
        int((~detected).sum()),
        len(set(bands)),
        reference,
    )
    return transient
