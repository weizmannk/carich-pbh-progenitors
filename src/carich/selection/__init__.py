"""Sample selection: host association, peak luminosity, first-look yields."""

from ._classes import HYDROGEN_POOR, group
from ._hosts import HostMatch, associate, read_crossmatch
from ._lightcurve import Peak, interpolate, peak, read_alerts
from ._luminosity import (
    band_extinction,
    bandpass,
    distance_modulus,
    distance_modulus_error,
)
from ._nickel import bolometric_correction, luminosity, nickel_mass
from ._select import select

__all__ = (
    "HYDROGEN_POOR",
    "HostMatch",
    "Peak",
    "associate",
    "band_extinction",
    "bandpass",
    "bolometric_correction",
    "distance_modulus",
    "distance_modulus_error",
    "group",
    "interpolate",
    "luminosity",
    "nickel_mass",
    "peak",
    "read_alerts",
    "read_crossmatch",
    "select",
)
