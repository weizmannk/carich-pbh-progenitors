import numpy as np
import pytest
from astropy import units as u
from dust_extinction.parameter_averages import G23

from .._luminosity import band_extinction, distance_modulus


def test_band_extinction_of_a_narrow_band_is_the_curve():
    wavelength = np.linspace(6390, 6410, 5) * u.AA
    expected = 3.1 * 0.1 * G23(Rv=3.1)(6400 * u.AA)
    assert band_extinction(wavelength, np.ones(5), Ebv=0.1) == pytest.approx(
        expected, rel=1e-3
    )


def test_band_extinction_is_linear_in_reddening_for_a_narrow_band():
    wavelength = np.linspace(4790, 4810, 5) * u.AA
    one = band_extinction(wavelength, np.ones(5), Ebv=0.05)
    two = band_extinction(wavelength, np.ones(5), Ebv=0.10)
    assert two == pytest.approx(2 * one, rel=1e-3)


def test_distance_modulus_matches_the_hubble_law_nearby():
    # 10 Mpc: mu = 30; Planck15 has H0 = 67.74 km/s/Mpc.
    assert distance_modulus(67.74 * 10 / 299792.458) == pytest.approx(30, abs=0.01)
