import numpy as np

from .._lightcurve import interpolate, peak

TIME = np.array([0.0, 3.0, 6.0, 9.0, 12.0])
ERROR = np.full(5, 0.05)


def test_peak_is_bracketed_when_seen_rising_and_falling():
    result = peak(TIME, [19.0, 18.2, 18.0, 18.4, 19.1], ERROR)
    assert result.time == 6.0
    assert result.magnitude == 18.0
    assert result.bracketed


def test_peak_is_not_bracketed_on_a_monotonic_decline():
    assert not peak(TIME, [18.0, 18.3, 18.7, 19.0, 19.4], ERROR).bracketed


def test_peak_is_not_bracketed_by_points_within_the_noise():
    assert not peak(TIME, [18.1, 18.05, 18.0, 18.05, 18.1], ERROR).bracketed


def test_peak_of_an_empty_light_curve():
    assert peak([], [], []) is None


def test_interpolate():
    magnitude = [19.0, 18.0, 18.0, 18.6, 19.2]
    assert interpolate(TIME, magnitude, 10.5, max_gap=5) == 18.9
    # Gap too long: fall back on a detection within two days, or give up.
    assert interpolate(TIME, magnitude, 10.5, max_gap=2) == 18.6
    assert np.isnan(interpolate([0.0, 20.0], [18.0, 19.0], 10.0, max_gap=5))
    assert np.isnan(interpolate(TIME, magnitude, 13.0, max_gap=5))
