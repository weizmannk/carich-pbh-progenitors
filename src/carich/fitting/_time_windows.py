"""Single source of truth for timeshift / time-window arithmetic.

Generic, with no NMMA dependency, so it's also usable from redback/mosfit.

Timeshift representations accepted throughout:
    None                        -> no shift
    float                       -> fixed shift
    {"min": a, "max": b}        -> uniform prior
    {"mu": m, "sigma": s}       -> Gaussian prior (centered on m)
    {"peak": p}                 -> delta function
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple, Union

Timeshift = Optional[Union[float, Dict[str, float]]]


def collapse_timeshift(ts: Timeshift) -> float:
    """Collapse a timeshift representation to a single representative float.

    Uses the lower bound for a uniform dict (min), the center for a
    Gaussian (mu), the peak for a fixed delta.
    """
    if ts is None:
        return 0.0
    if isinstance(ts, dict):
        if "min" in ts:
            return float(ts["min"])
        if "peak" in ts:
            return float(ts["peak"])
        if "mu" in ts:
            return float(ts["mu"])
        return 0.0
    return float(ts)


def timeshift_bounds(ts: Timeshift) -> Tuple[float, float]:
    """Return (lower bound, upper bound) the timeshift prior can produce."""
    if ts is None:
        return 0.0, 0.0
    if isinstance(ts, dict):
        lo = float(ts.get("min", ts.get("mu", ts.get("peak", 0.0))))
        hi = float(ts.get("max", ts.get("mu", ts.get("peak", 0.0))))
        return lo, hi
    val = float(ts)
    return val, val


def em_window(data_tmin: float, data_tmax: float, ts: Timeshift) -> Tuple[float, float]:
    """Model-grid window [em_tmin, em_tmax] covering the full timeshift prior.

    nmma-1.0's check_model_time_consistency requires the EM window to cover
    every shifted model start:
      - timeshift = hi > 0 shifts the model forward -> em_tmin decreases by hi
      - timeshift = lo < 0 shifts the model backward -> em_tmax increases by |lo|
    """
    lo, hi = timeshift_bounds(ts)
    em_tmin = float(data_tmin) - max(hi, 0.0)
    em_tmax = float(data_tmax) + max(-lo, 0.0)
    logging.debug(
        "em_window: [%.2f, %.2f]  (data [%.2f, %.2f], timeshift [%.2f, %.2f])",
        em_tmin,
        em_tmax,
        data_tmin,
        data_tmax,
        lo,
        hi,
    )
    return em_tmin, em_tmax


def data_window(
    tmin: Optional[float], tmax: Optional[float], ts: Timeshift
) -> Tuple[Optional[float], Optional[float]]:
    """Data-selection window [tmin_adj, tmax_adj] relative to the trigger."""
    ts_val = collapse_timeshift(ts)
    if ts_val < 0:
        return tmin, (tmax + abs(ts_val)) if tmax is not None else None
    return (tmin - ts_val) if tmin is not None else None, tmax
