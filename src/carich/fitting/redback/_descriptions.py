"""Human-readable model metadata, for the Astro-COLIBRI picker and frontend.

Nothing here reaches redback. These are labels and prose for a user interface:
the fitting code reads parameters, priors and constraints from redback itself
(see ``_models``), never from this file.

The field shape matches what the Astro-COLIBRI picker expects
(``PICKER_MODELS`` in the Redback API routes), so wiring a model into the
frontend is a copy rather than a rewrite.

The caveats were written from the source of redback's constraint functions
(``redback.constraints``), not from the published papers, so they describe what
the code actually enforces. Where the two disagree, the code is what runs.
"""

from __future__ import annotations

# Unit and meaning of the parameters whose name does not carry them. redback's
# priors have latex labels but no machine-readable units, and these are the keys
# a reader of a posterior table would otherwise have to guess.
PARAMETER_NOTES: dict[str, str] = {
    "f_nickel": "Ni-56 fraction of the ejecta; M_Ni = f_nickel * mej",
    "mej": "ejecta mass, Msun",
    "vej": "ejecta velocity, km/s",
    "kappa": "optical opacity, cm^2/g",
    "kappa_gamma": "gamma-ray opacity, cm^2/g",
    "temperature_floor": "blackbody temperature floor, K",
    "p0": "initial spin period, ms",
    "bp": "dipole magnetic field, 1e14 G",
    "mass_ns": "neutron star mass, Msun",
    "theta_pb": "magnetic obliquity, rad",
    "log10_mass": "extended material mass, log10 Msun",
    "log10_radius": "extended material radius, log10 cm",
    "log10_energy": "energy deposited by the shock, log10 erg",
    "nn": "outer density slope",
    "delta": "inner density slope",
    "l0": "bolometric luminosity at 1 s, erg/s",
    "t_0_turn": "fallback turn-on time, days",
    "mbh_6": "black hole mass, 1e6 Msun",
    "stellar_mass": "disrupted star mass, Msun",
    "eta": "accretion feedback efficiency",
    "alpha": "viscosity parameter",
    "beta": "tidal impact parameter",
    "redshift": "cosmological redshift",
    "t0": "explosion time, MJD (sampled; redback's own parameterisation)",
    "av_mw": "Milky Way V-band extinction, mag",
    "av_host": "host-galaxy V-band extinction, mag",
    "peak_time_mjd": "epoch of the template's maximum light, MJD",
    "x1": "SALT light-curve shape (stretch)",
    "c": "SALT colour",
    "peak_abs_mag": "peak absolute magnitude, AB, in standard::b",
    "ebv": "host-galaxy E(B-V), mag",
}

DESCRIPTIONS: dict[str, dict] = {
    "arnett": {
        "label": "SN - Arnett (Ni-56 powered)",
        "description": (
            "Radioactively powered supernova, Ni-56 to Co-56 to Fe-56 decay "
            "heating smoothed by Arnett photon diffusion through a "
            "homologously expanding ejecta shell with gamma-ray leakage, "
            "emitted as a cooling blackbody with a temperature floor; the "
            "baseline for any Ni-56 powered transient, with no spectral-class "
            "assumption."
        ),
        "derived": ("M_Ni = f_nickel * mej", "L_bol", "R_ph", "T_eff"),
        "reference": "Arnett 1982",
        "reference_url": "https://ui.adsabs.harvard.edu/abs/1982ApJ...253..785A",
        "caveat": (
            "M_Ni is robust; mej is prior-dependent without a 56-Co tail and "
            "degenerate with kappa. Constraint (nuclear_burning_constraints, "
            "emax_constraint): the Ni-56 burning energy must exceed the "
            "kinetic energy."
        ),
    },
    "slsn": {
        "label": "SLSN - magnetar powered",
        "description": (
            "Superluminous supernova, dipole spin-down of a millisecond "
            "magnetar (flat, then proportional to t^-2) smoothed by Arnett "
            "diffusion with gamma-ray leakage, emitted as a cooling blackbody "
            "cut off below 3000 A."
        ),
        "derived": ("spin-down timescale", "E_rot", "L_bol", "R_ph"),
        "reference": "Nicholl, Guillochon & Berger 2017",
        "reference_url": "https://ui.adsabs.harvard.edu/abs/2017ApJ...850...55N",
        "caveat": (
            "Constraint (slsn_constraint, via get_priors(constraint=True)): "
            "E_rot = 2.6e52 (M_ns/1.4)^1.5 p0^-2 must exceed E_kin + 1e51 erg "
            "of neutrino losses, and the nebular phase must not begin before "
            "100 d. Unreliable past that phase."
        ),
    },
    "shock_cooling": {
        "label": "Shock cooling of extended material",
        "description": (
            "Shock-cooling emission, low-mass extended material with a "
            "broken power-law density profile radiates the energy deposited "
            "by the shock, falling as (t_d/t)^(4/(n-2)) until the diffusion "
            "time and exponentially after, from an analytically tracked "
            "photosphere. Covers the first hours to days only."
        ),
        "derived": ("t_d diffusion time", "L_bol", "R_ph", "T_eff"),
        "reference": "Piro, Haynie & Yao 2021",
        "reference_url": "https://ui.adsabs.harvard.edu/abs/2021ApJ...909..209P",
        "caveat": (
            "Magnitude grid stops at 30 d observer frame: later data would be "
            "extrapolation. No Ni-56 peak, so use shock_cooling_and_arnett for "
            "a full light curve. redback samples nn and delta where the paper "
            "fixes them at n ~ 10 and delta ~ 1.1 (Chevalier & Soker 1989); "
            "priors_redback pins them. The only one of these five models with "
            "no physical constraint, and the only one redback ships no "
            "BUILTIN_MODEL_METADATA for."
        ),
    },
    "shock_cooling_and_arnett": {
        "label": "SN - shock cooling + Arnett",
        "description": (
            "Both peaks of a stripped-envelope supernova in one model, the "
            "shock-cooling emission of low-mass extended material summed with "
            "the Ni-56 powered Arnett peak; the luminosities add, so the "
            "early decline and the radioactive maximum are fitted jointly "
            "rather than over separate time windows. The union of "
            "shock_cooling's and arnett's parameters."
        ),
        "derived": ("M_Ni = f_nickel * mej", "t_d diffusion time", "L_bol", "R_ph"),
        "reference": "Piro, Haynie & Yao 2021 + Arnett 1982",
        "reference_url": "https://ui.adsabs.harvard.edu/abs/2021ApJ...909..209P",
        "caveat": (
            "The model for a Type IIb, whose first peak is shock cooling and "
            "not nickel: fitting arnett alone to such a light curve drives t0 "
            "against its lower bound and biases M_Ni. Twelve free parameters "
            "against the five of arnett, so it needs the early epochs to be "
            "covered or the shock-cooling half is unconstrained. redback ships "
            "no BUILTIN_MODEL_METADATA for it, so its family comes from "
            "_MODEL_TYPE_FALLBACK, and no constraint either: unlike arnett, "
            "the Ni-56 burning energy is NOT checked against the kinetic "
            "energy here. priors_redback pins nn and delta as it does for "
            "shock_cooling."
        ),
    },
    "tde_analytical": {
        "label": "TDE - fallback powered",
        "description": (
            "Tidal disruption event, a bolometric luminosity held flat "
            "until a turn-on time then decaying as t^-5/3, smoothed by Arnett "
            "diffusion through an expanding shell and emitted as a blackbody "
            "cut off below 3000 A."
        ),
        "derived": ("radiated energy", "L_bol", "R_ph", "T_bb"),
        "reference": "Sarin & Metzger 2024",
        "reference_url": "https://ui.adsabs.harvard.edu/abs/2024ApJ...961L..19S",
        "caveat": (
            "Phenomenological: l0 is a luminosity normalisation, not an "
            "energy, and the fit yields no black hole or stellar mass. Use "
            "cooling_envelope for those. Constraint "
            "(simple_fallback_constraints, en_constraint): the fallback "
            "energy l0 * 5/2 / (t_0_turn)^(2/3) plus 1e51 erg must exceed the "
            "kinetic energy, and the nebular phase must not begin before "
            "100 d. Its kappa prior runs to 100 cm^2/g, far wider than the "
            "other models' 2."
        ),
    },
    "sncosmo_models": {
        "label": "Template - sncosmo library",
        "description": (
            "Empirical spectral template, an observed supernova's own time "
            "series of spectra, redshifted and integrated through the "
            "bandpasses; no physics is assumed and nothing is derived, which "
            "is what makes it the right choice when a spectral class is known "
            "and a parametric model would be fitted outside its domain. The "
            "template is chosen with --sncosmo-model, so this one entry covers "
            "the whole sncosmo library."
        ),
        "derived": ("peak apparent magnitude per band", "rise and decline rate"),
        "reference": "Guillochon et al. 2017, sncosmo",
        "reference_url": "https://ui.adsabs.harvard.edu/abs/2017ApJ...835...64G",
        "caveat": (
            "No ejecta mass, no nickel mass, no explosion energy: a template "
            "says what the light curve looks like, not what powered it, so "
            "M_Ni is not available from this model. Two epochs are sampled, "
            "t0 the explosion and peak_time_mjd maximum light, the second "
            "reduced to days since the first by t0_base_model's own "
            "peak_time_mjd branch. The template is valid only over its phase "
            "range (-2.5 to 223 d rest frame for v19-1993j-corr), and its "
            "brightness is set through peak_abs_mag rather than the raw "
            "normalisation, whose meaning and scale differ per template. "
            "A SALT template also brings x1 and c, on redback's own "
            "salt2.prior values. Extinction comes from two separate keys: ebv, "
            "the host term redback gives sncosmo's rest-frame CCM89 effect, "
            "and av_mw, applied per band with fitzpatrick99 around the "
            "template because redback registers sncosmo's F99 effect without "
            "ever setting mwebv. Omitting av_mw does not remove Galactic "
            "reddening, it transfers it to the host key."
        ),
    },
    "cooling_envelope": {
        "label": "TDE - cooling envelope",
        "description": (
            "Tidal disruption event, debris circularises into a hot "
            "pressure-supported envelope radiating near the Eddington limit "
            "while contracting on the Kelvin-Helmholtz timescale, giving a "
            "rising temperature and an optical decline that mimics fallback; "
            "accretion onto the black hole is regulated by a viscous "
            "timescale and a feedback efficiency."
        ),
        "derived": (
            "envelope radius R_v",
            "T_eff",
            "R_ph",
            "late-time plateau",
            "X-ray turn-on delay",
        ),
        "reference": "Metzger 2022",
        "reference_url": "https://ui.adsabs.harvard.edu/abs/2022ApJ...937L..12M",
        "caveat": (
            "t0 is the start of the post-circularisation phase, not the "
            "disruption: the model's internal origin is time_since_fb, so its "
            "t0 can legitimately postdate the first detection. Valid only "
            "after circularisation (t >~ t_fb); use "
            "gaussianrise_cooling_envelope or bpl_cooling_envelope to include "
            "the rise. f_debris is an optional kwarg (default 1.0, or derived "
            "from beta with calculate_f_debris=True), not a prior parameter. "
            "Constraint (cooling_envelope_constraints): eta > 0.01 "
            "M_*^(-7/15) mbh_6^(2/3) and beta < 12 M_*^(7/15) mbh_6^(-2/3). "
            "Its 'spectra' output reports time in SECONDS where every other "
            "model reports days, so it carries no sampled extinction and its "
            "evidence is not comparable with the models that do."
        ),
    },
}


def describe(name: str) -> str:
    """Prose summary for the frontend, joined with redback's live parameters."""
    from ._models import constraint_keys, extinction_model, sampled

    entry = DESCRIPTIONS[name]
    parameters = ", ".join(
        f"{key} [{PARAMETER_NOTES[key]}]" if key in PARAMETER_NOTES else key
        for key in sampled(name)
    )
    return (
        f"{entry['label']} ({entry['reference']})\n"
        f"{entry['description']}\n"
        f"Sampled: {parameters}\n"
        f"Derived: {', '.join(entry['derived'])}\n"
        f"Extinction: {extinction_model(name) or 'none, no usable wrapper'}\n"
        f"Constraints: {', '.join(sorted(constraint_keys(name))) or 'none'}\n"
        f"Caveat: {entry['caveat']}"
    )
