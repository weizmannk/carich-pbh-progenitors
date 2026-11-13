import numpy as np
from astropy import units as u

_L0 = 3.0128e28 * u.W
"""Luminosity of a source of zero absolute bolometric magnitude (IAU 2015)."""


def bolometric_correction(color: float) -> float:
    """
    Bolometric correction to the g band from the g-r color.

    Parameters
    ----------
    color
        Color :math:`g-r`, corrected for extinction.

    Returns
    -------
    :
        :math:`M_\\mathrm{bol} - M_g`. NaN outside the range of validity,
        :math:`-0.3 < g-r < 1.0`.

    Notes
    -----
    This is the fit of [1]_ for stripped-envelope supernovae in the
    radioactively powered phase. It was not calibrated on
    Type Ia supernovae or on calcium-rich transients, whose spectra turn
    nebular within weeks; treat the result as a first look.

    References
    ----------
    .. [1] Lyman, Bersier & James 2014, MNRAS, 437, 3848.

    Examples
    --------
    >>> from carich.selection import bolometric_correction
    >>> round(bolometric_correction(0.5), 3)
    -0.223
    """
    if not -0.3 < color < 1.0:
        return np.nan
    return 0.054 - 0.195 * color - 0.719 * color**2


def luminosity(magnitude: float) -> u.Quantity[u.physical.power]:
    """
    Luminosity of a source of given absolute bolometric magnitude.

    Parameters
    ----------
    magnitude
        Absolute bolometric magnitude.

    Examples
    --------
    >>> from carich.selection import luminosity
    >>> luminosity(-16).to("erg / s").round(-39)
    <Quantity 7.57e+41 erg / s>
    """
    return (_L0 * 10 ** (-0.4 * magnitude)).to(u.erg / u.s)


def nickel_mass(
    peak_luminosity: u.Quantity[u.physical.power],
    rise_time: u.Quantity[u.physical.time],
) -> u.Quantity[u.physical.mass]:
    """
    Mass of nickel-56 from Arnett's rule.

    Parameters
    ----------
    peak_luminosity
        Bolometric luminosity at maximum light.
    rise_time
        Time from explosion to maximum light.

    Notes
    -----
    Arnett's rule equates the peak luminosity to the instantaneous power of
    the nickel-56 and cobalt-56 decay chain at that time. It holds to a few
    tens of percent for centrally concentrated nickel and constant opacity
    [1]_. The coefficients are those of [2]_. The rise time is rarely measured from
    alert photometry alone, so quote a range of plausible rise times rather
    than a single value.

    References
    ----------
    .. [1] Khatami & Kasen 2019, ApJ, 878, 56.
    .. [2] Stritzinger & Leibundgut 2005, A&A, 431, 423.

    Examples
    --------
    >>> from astropy import units as u
    >>> from carich.selection import nickel_mass
    >>> nickel_mass(1e42 * u.erg / u.s, 12 * u.day).round(3)
    <Quantity 0.034 solMass>
    """
    t = rise_time.to_value(u.day)
    power = 6.45e43 * np.exp(-t / 8.8) + 1.45e43 * np.exp(-t / 111.3)
    return (peak_luminosity.to_value(u.erg / u.s) / power) * u.Msun
