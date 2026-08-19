"""Milky Way extinction correction (E(B-V)) via dustmaps.

Shared across all fitting backends (nmma, redback, mosfit): the MW
correction depends on the object's coordinates, not on the fitting tool.

get_planck_ebv (Planck GNILC) is the actual MW dereddening value used by
redback/mosfit. get_sfd_ebv (raw SFD98) is separate: nmma uses it only to
bound the free Ebv nuisance parameter (--Ebv-max), scaled by the Schlafly
& Finkbeiner (2011) recalibration factor, a correction specific to SFD and
not applicable to GNILC. The two maps disagree by ~10-15% at a given line
of sight (SF2011 corrects a global SFD calibration bias, while GNILC
corrects spatially-varying CIB contamination: different systematics, with
no reason to converge pointwise), which is fine here since Ebv-max is only
a prior bound, not a measurement.

Reference: Planck Collaboration 2016, A&A 596, A109 (arXiv:1605.09387),
GNILC dust optical depth map, E(B-V) = tau_353 x 1.49e-4 mag.
"""

import warnings
from functools import cache

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.utils.data import download_file

with warnings.catch_warnings():
    # Bypassing dustmaps' own config/fetch mechanism (see _dust_map below),
    # so its "no config file" warning does not apply here.
    warnings.simplefilter("ignore", UserWarning)
    import dustmaps.sfd
    from dustmaps.planck import PlanckGNILCQuery

# Download/cache pattern adapted from m4opt (github.com/m4opt/m4opt).
_GNILC_SOURCES = [
    "https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/component-maps/foregrounds/COM_CompMap_Dust-GNILC-Model-Opacity_2048_R2.01.fits",
    "https://pla.esac.esa.int/pla/aio/product-action?MAP.MAP_ID=COM_CompMap_Dust-GNILC-Model-Opacity_2048_R2.01.fits",
]


@cache
def _dust_map() -> PlanckGNILCQuery:
    """Download (IRSA, falling back to ESA) and cache the GNILC opacity map."""
    path = download_file(_GNILC_SOURCES[-1], cache=True, sources=_GNILC_SOURCES)
    return PlanckGNILCQuery(path)


def get_planck_ebv(ra: float, dec: float) -> float:
    """E(B-V) at (ra, dec) in degrees (ICRS), from the Planck GNILC map."""
    coords = SkyCoord(ra=ra * u.degree, dec=dec * u.degree, frame="icrs")
    return float(_dust_map()(coords))


# Schlafly & Finkbeiner (2011) recalibration factor for SFD-family E(B-V).
EBV_SF2011 = 0.86


def get_sfd_ebv(ra: float, dec: float) -> float:
    """Raw SFD98 E(B-V) at (ra, dec) in degrees (ICRS), no recalibration.

    Only used (by nmma-fit) to bound the free Ebv nuisance parameter; apply
    EBV_SF2011 separately if needed.
    """
    coords = SkyCoord(ra=ra * u.degree, dec=dec * u.degree, frame="icrs")
    try:
        return float(dustmaps.sfd.SFDQuery()(coords))
    except FileNotFoundError:
        dustmaps.sfd.fetch()
        return float(dustmaps.sfd.SFDQuery()(coords))
