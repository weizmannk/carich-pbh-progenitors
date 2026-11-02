#!/usr/bin/env bash
# SN2021ugl, Type IIb. Copy any block below, or run this file.
#
# Every parameter is written out, including those that have a default, so a run
# can be read without consulting --help. The defaults are:
#   --nlive 2048   --sampler pymultinest   --seed 42
#   --upper-limit-sigma 3.0   --plot-theme print
#   --priors              resolved automatically: priors_redback/<model>.prior,
#                         or sncosmo_salt.prior / sncosmo_generic.prior on the
#                         sncosmo path, by whether the template exposes x0.
#                         Written out below so a run reads without that rule.
#   --t0-min / --t0-max   from redback_models.T0_WINDOW, per model:
#                         (-2, +1) for the physical models, bounding t0;
#                         (-30, +30) for sncosmo_models, bounding peak_time
#
# Lower --nlive to check a setup: pymultinest fits the 4-parameter template in
# ~7 s at nlive 200. dynesty is ~10x slower for the same fit.
# --plot-theme astrocolibri gives the branded figure instead of the paper one.
#
# Extinction. --ra/--dec let the fit query the Planck GNILC dust map itself and
# bound av_mw at R_V x E(B-V); pass --ebv-mw to set that bound by hand instead.
# Both apply only to the physical models: the sncosmo path has no Galactic term
# at all, because redback adds sncosmo's F99 effect but never sets mwebv.
#
# Host extinction has no map to bound it, so it is off unless --sample-av-host
# is given, with --av-host-max as its ceiling in magnitudes of A_V. On the
# sncosmo path it arrives as the 'ebv' key, converted by A_V = R_V x E(B-V), so
# --av-host-max 1.0 gives ebv ~ Uniform(0, 0.3226). Leaving it off pins it at 0
# deliberately: redback's own default for that key is 0.1, which would redden
# every template silently.
#
# redshift is sampled: there is no --z-known here, so it runs on
# Uniform(0.01, 3) from the prior file, truncated on the sncosmo path at the
# template's spectral validity for the bands in the data. It is degenerate with
# the brightness, so the fit can prefer a redshift far from the spectroscopic
# one; pass --z-known to pin it.

cd /Users/rk281962/Projects/CBC/REDBACK/carich-pbh-progenitors

# pymultinest loads libmultinest at import time. Calling .venv-redback/bin/python
# directly skips the venv's activate script, which is what exports this, and the
# fit then dies at the first likelihood call with
# "dlsym(RTLD_DEFAULT, run): symbol not found". Note that running anything under
# /usr/bin/time strips it again: macOS SIP clears DYLD_* for protected binaries.
export DYLD_LIBRARY_PATH="$HOME/.local/lib:$DYLD_LIBRARY_PATH"


# Shock cooling + Arnett, the two peaks from physics. Gives M_Ni.
# 12 free parameters, so the slowest of the three. kappa is left on redback's
# own prior, Uniform(0.05, 2); see the --kappa variant at the end.
.venv-redback/bin/python src/fitting/redback-fit/fit_redback.py \
  --object SN2021ugl \
  --data data/processed/SN2021ugl.dat \
  --model shock_cooling_and_arnett \
  --priors priors_redback/shock_cooling_and_arnett.prior \
  --t0-min -2.0 \
  --t0-max 1.0 \
  --upper-limit-sigma 3.0 \
  --nlive 2048 \
  --sampler pymultinest \
  --seed 42 \
  --plot-theme print \
  --sample-av-host \
  --av-host-max 1.0 \
  --ra 246.9820265 \
  --dec 20.253166509091 \
  --outdir results/redback


# SN 1993J template, the Type IIb prototype, on all the data.
#
# --t0-min/--t0-max bound peak_time here, not t0: this path never reads t0
# (measured, two calls differing only in t0 return identical magnitudes), and
# peak_time is the epoch of the template's phase zero, i.e. maximum light.
# The window has to be wide because the template's maximum need not sit at the
# brightest observed point: on a double-peaked IIb it belongs at the first,
# shock-cooling peak, and forcing it onto the Ni-56 peak 18 days later pushes
# the early data outside the template (ln Z -1723 against -32).
#
# 4 free parameters: redshift, peak_time, amplitude, ebv.
.venv-redback/bin/python src/fitting/redback-fit/fit_redback.py \
  --object SN2021ugl \
  --data data/processed/SN2021ugl.dat \
  --model sncosmo_models \
  --sncosmo-model v19-1993j-corr \
  --priors priors_redback/sncosmo_generic.prior \
  --t0-min -30.0 \
  --t0-max 30.0 \
  --upper-limit-sigma 3.0 \
  --nlive 2048 \
  --sampler pymultinest \
  --seed 42 \
  --plot-theme print \
  --sample-av-host \
  --av-host-max 1.0 \
  --outdir results/redback


# SALT3, same interface, as a control: it is a Type Ia template, so a worse fit
# here is the expected answer for a IIb. Takes x0, x1, c instead of amplitude,
# so 6 free parameters, on redback's own priors/salt2.prior values.
.venv-redback/bin/python src/fitting/redback-fit/fit_redback.py \
  --object SN2021ugl \
  --data data/processed/SN2021ugl.dat \
  --model sncosmo_models \
  --sncosmo-model salt3 \
  --priors priors_redback/sncosmo_salt.prior \
  --t0-min -30.0 \
  --t0-max 30.0 \
  --upper-limit-sigma 3.0 \
  --nlive 2048 \
  --sampler pymultinest \
  --seed 42 \
  --plot-theme print \
  --sample-av-host \
  --av-host-max 1.0 \
  --outdir results/redback
