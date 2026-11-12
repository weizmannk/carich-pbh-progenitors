import numpy as np
from astropy import constants as c
from astropy import units as u
from astropy.cosmology import Planck15
from dust_extinction.parameter_averages import G23


def distance_modulus(redshift: float) -> float:
    """
    Distance modulus in the Planck 2015 cosmology.

    Parameters
    ----------
    redshift
        Redshift of the transient.

    Examples
    --------
    >>> from carich.selection import distance_modulus
    >>> round(distance_modulus(0.03), 2)
    35.66
    """
    return float(Planck15.distmod(redshift).value)


def distance_modulus_error(
    redshift: float, velocity: u.Quantity[u.physical.velocity] = 300 * u.km / u.s
) -> float:
    """
    Uncertainty on the distance modulus from a peculiar velocity.

    Parameters
    ----------
    redshift
        Redshift of the transient.
    velocity
        Peculiar velocity along the line of sight.

    Examples
    --------
    >>> from carich.selection import distance_modulus_error
    >>> round(distance_modulus_error(0.02), 2)
    0.11
    """
    return float(5 / np.log(10) * (velocity / c.c).to_value(u.one) / redshift)


def bandpass(name: str):
    """
    Wavelength and transmission of a bandpass known to :mod:`sncosmo`.

    Parameters
    ----------
    name
        Name of the bandpass, e.g. ``"ztfg"`` or ``"atlaso"``.

    Returns
    -------
    wavelength : :class:`astropy.units.Quantity`
        Wavelength grid.
    transmission : :class:`numpy.ndarray`
        Dimensionless transmission on that grid.
    """
    import sncosmo

    band = sncosmo.get_bandpass(name)
    return band.wave * u.AA, band.trans


def band_extinction(
    wavelength: u.Quantity[u.physical.length],
    transmission,
    Ebv: float,
    Rv: float = 3.1,
) -> float:
    """
    Milky Way extinction in a bandpass.

    Parameters
    ----------
    wavelength
        Wavelength grid of the bandpass.
    transmission
        Transmission on that grid.
    Ebv
        Reddening color excess, :math:`E(B-V)`.
    Rv
        Ratio of total to selective extinction.

    Returns
    -------
    :
        Extinction in magnitudes.

    Notes
    -----
    The extinction curve is :class:`dust_extinction.parameter_averages.G23`,
    as in M4OPT and NMMA, applied in the observer frame and averaged over the
    bandpass for a source with a flat photon spectrum.

    Pass the reddening of the Planck GNILC map as it is. Only values read from
    the Schlegel, Finkbeiner & Davis (1998) map need the factor 0.86 of
    Schlafly & Finkbeiner (2011); applying it to GNILC values would
    underestimate the extinction by 14%.

    Examples
    --------
    >>> import numpy as np
    >>> from astropy import units as u
    >>> from carich.selection import band_extinction
    >>> wavelength = np.linspace(5450, 5550, 11) * u.AA
    >>> round(band_extinction(wavelength, np.ones(11), Ebv=0.1), 2)
    0.31
    """
    throughput = G23(Rv=Rv).extinguish(wavelength, Ebv=Ebv)
    weight = transmission * wavelength.value
    return float(
        -2.5
        * np.log10(
            np.trapezoid(weight * throughput, wavelength.value)
            / np.trapezoid(weight, wavelength.value)
        )
    )
