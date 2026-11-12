GROUPS = {
    "Ia": "Ia",
    "Ia-norm": "Ia",
    "Ia-91T": "Ia",
    "Ia-pec": "Ia",
    "Ia-03fg": "Ia",
    "Ia-CSM": "Ia",
    "Ia-91bg": "91bg",
    "Ia-02cx": "Iax",
    "Ca-rich": "Ca-rich",
    "Ib": "Ibc",
    "Ic": "Ibc",
    "Ib/c": "Ibc",
    "Ib-norm": "Ibc",
    "Ib-pec": "Ibc",
    "Ic-BL": "Ibc",
    "Ibn": "Ibc",
    "Icn": "Ibc",
    "Type I": "Ibc",
    "Type II": "II",
    "IIP": "II",
    "IIL": "II",
    "IIb": "II",
    "IIn": "II",
    "II-pec": "II",
}
"""Spectroscopic classifications, as spelled on Fritz, and their group."""

HYDROGEN_POOR = frozenset({"Ia", "91bg", "Iax", "Ca-rich", "Ibc"})
"""Groups kept for the analysis.

Type Ib/c is kept on purpose: most calcium-rich transients are classified as
Type Ib/c at peak and are only recognized from their nebular spectra.
"""


def group(classification: str) -> str:
    """
    Group of a spectroscopic classification.

    Parameters
    ----------
    classification
        Classification as spelled on Fritz, e.g. ``"Ia-91bg"``.

    Returns
    -------
    :
        One of ``"Ia"``, ``"91bg"``, ``"Iax"``, ``"Ca-rich"``, ``"Ibc"``,
        ``"II"``, or ``"other"`` for anything that is not a supernova.

    Examples
    --------
    >>> from carich.selection import group
    >>> group("Ia-02cx")
    'Iax'
    >>> group("Classical Nova")
    'other'
    """
    return GROUPS.get(str(classification).strip(), "other")
