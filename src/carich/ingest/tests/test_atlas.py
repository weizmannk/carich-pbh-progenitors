import numpy as np
import pytest

from .._atlas import clean, read, stack, write

TEXT = """###MJD m dm uJy duJy F err chi/N RA Dec x y maj min phi apfit mag5sig Sky Obs
59400.30 18.9 0.1 100 20 o 0 1.0 10 20 0 0 2 2 0 -0.4 19.5 20.0 a
59400.31 18.9 0.1 120 20 o 0 1.1 10 20 0 0 2 2 0 -0.4 19.5 20.0 b
59400.32 18.9 0.1 110 500 o 0 1.0 10 20 0 0 2 2 0 -0.4 19.5 20.0 c
59400.33 18.9 0.1 110 20 o 0 50.0 10 20 0 0 2 2 0 -0.4 19.5 20.0 d
59401.30 99.0 9.9 5 20 o 0 1.0 10 20 0 0 2 2 0 -0.4 19.5 20.0 e
59401.31 99.0 9.9 -5 20 o 0 1.0 10 20 0 0 2 2 0 -0.4 19.5 20.0 f
59401.40 18.5 0.1 150 30 c 0 1.0 10 20 0 0 2 2 0 -0.4 19.5 20.0 g
"""


def test_clean_drops_noisy_and_poorly_fit_exposures():
    table = clean(read(TEXT))
    assert list(table["Obs"]) == ["a", "b", "e", "f", "g"]


def test_stack_averages_each_night_and_filter():
    nights = stack(clean(read(TEXT)))
    assert list(nights["band"]) == ["atlaso", "atlaso", "atlasc"]

    first = nights.iloc[0]
    error = 20 / np.sqrt(2)
    assert first["mag"] == pytest.approx(23.9 - 2.5 * np.log10(110))
    assert first["magerr"] == pytest.approx(2.5 / np.log(10) * error / 110)


def test_stack_reports_a_limit_for_a_non_detection():
    limit = stack(clean(read(TEXT))).iloc[1]
    assert np.isinf(limit["magerr"])
    assert limit["mag"] == pytest.approx(23.9 - 2.5 * np.log10(3 * 20 / np.sqrt(2)))


def test_write_merges_with_an_existing_file(tmp_path):
    path = tmp_path / "object.dat"
    path.write_text("2021-07-24T05:25:30.996 ztfg 18.8802 inf\n")
    write(stack(clean(read(TEXT))), path, append=True)
    lines = path.read_text().splitlines()
    assert len(lines) == 4
    assert lines == sorted(lines)
    assert lines[-1].split()[1] == "ztfg"
