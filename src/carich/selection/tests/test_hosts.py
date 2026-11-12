import numpy as np
import pandas as pd

from .._hosts import associate

MATCHES = pd.DataFrame(
    {
        "clu_id": [1, 2],
        "clu_z": [0.0048, 0.0262],
        "clu_dist_kpc": [77.0, 21.0],
        "clu_sep_arcsec": [800.0, 40.0],
    }
)


def test_associate_picks_the_closest_redshift():
    host = associate(MATCHES, 0.0266)
    assert host.clu_id == 2
    assert np.isclose(host.dz, 0.0004)
    assert host.offset == 21.0
    assert not host.copied


def test_associate_flags_a_copied_redshift():
    assert associate(MATCHES, 0.0262).copied


def test_associate_needs_a_redshift_and_a_galaxy():
    assert associate(MATCHES, np.nan) is None
    assert associate(MATCHES.iloc[:0], 0.02) is None
