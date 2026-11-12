import numpy as np
import pytest
from astropy import units as u

from .._nickel import bolometric_correction, luminosity, nickel_mass


def test_bolometric_correction_range():
    assert np.isnan(bolometric_correction(1.2))
    assert np.isnan(bolometric_correction(np.nan))
    assert bolometric_correction(0.0) == 0.054


def test_luminosity_of_the_sun():
    assert luminosity(4.74).to_value(u.Lsun) == pytest.approx(1, rel=0.01)


def test_nickel_mass_of_a_normal_type_ia():
    # 0.6 Msun of nickel peaking after 19 days gives about 1.2e43 erg/s.
    mass = nickel_mass(1.2e43 * u.erg / u.s, 19 * u.day)
    assert mass.to_value(u.Msun) == pytest.approx(0.6, rel=0.05)


def test_nickel_mass_grows_with_rise_time():
    low, high = nickel_mass(1e42 * u.erg / u.s, [10, 15] * u.day)
    assert low < high
