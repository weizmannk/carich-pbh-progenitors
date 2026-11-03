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
# ~26 s at nlive 2048, against ~9 min for the same fit in NMMA (measured, same
# sampler and seed; the gap is NMMA's 401-point model grid against redback's 65
# observed epochs). dynesty is a further ~10x slower than pymultinest.
# --plot-theme astrocolibri gives the branded figure instead of the paper one.
#
# Each panel title carries chi2/d.o.f., the median of the posterior draws
# against the detections, so the figures read alongside the NMMA ones. It is a
# diagnostic computed after the fit, not the quantity the sampler maximised.
#
# Extinction. --ra/--dec let the fit query the Planck GNILC dust map itself and
# bound av_mw at R_V x E(B-V); pass --ebv-mw to set that bound by hand instead.
# The bound is marginalised, never imposed, which is the same treatment NMMA
# gives --use-Ebv --Ebv-max. On SN2021ugl the map allows E(B-V) = 0.0637 and
# the fit settles at 0.027 +/- 0.018, so dereddening the photometry with the
# map value instead would over-correct by about 0.08 mag in g.
#
# This now applies on the sncosmo path too. redback cannot carry it there --
# it registers sncosmo's F99 effect but never sets mwebv, and the parameter
# cannot be supplied either because model.update() runs before add_effect --
# so av_mw is applied per band with fitzpatrick99 around the template, exactly
# as NMMA does in apply_extinction_correction. Leaving it out does not drop
# the reddening, it hands it to the host term: measured here, 'ebv' reads
# 0.244 without av_mw and 0.214 with it, and was being reported as host
# extinction. Pass --ra/--dec on the sncosmo blocks or the two stay mixed.
#
# Host extinction has no map to bound it, so it is off unless --sample-av-host
# is given, with --av-host-max as its ceiling in magnitudes of A_V. On the
# sncosmo path it arrives as the 'ebv' key, converted by A_V = R_V x E(B-V), so
# --av-host-max 1.0 gives ebv ~ Uniform(0, 0.3226). Leaving it off pins it at 0
# deliberately: redback's own default for that key is 0.1, which would redden
# every template silently.
#
# Likelihood. --likelihood takes the three redback ships for this data mode.
# None of them does what NMMA does in one function, so the choice is a real
# trade-off and is left open rather than decided here:
#
#   upper-limits  GaussianLikelihoodWithUpperLimits, the default, and the one
#                 redback picks by itself. sigma = sigma_i. Keeps the 23
#                 non-detections through a survival term. NO error budget.
#   quadrature    GaussianLikelihoodQuadratureNoise. sqrt(sigma_i^2 + sigma^2),
#                 the formula NMMA applies behind --em-error-budget, except
#                 sigma is sampled here instead of given. No survival term.
#   systematic    GaussianLikelihoodWithSystematicNoise.
#                 sqrt(sigma_i^2 + model^2 sigma^2), a fractional budget. It
#                 was written for flux: in magnitudes 'model' is ~19, so sigma
#                 lands near 0.26 and means something else. No survival term.
#
# The two budget variants have no survival term, so they refuse to run while
# limits are present rather than dropping them quietly; --detections-only drops
# them on purpose. That makes their ln Z NOT comparable with the default run:
# 42 points against 65, and a sigma the other does not have. Measured here at
# nlive 512: -2.24 / +10.62 / -0.59 respectively, which is a change of dataset,
# not evidence that one likelihood fits better.
#
# --sigma-sys-max bounds the sampled sigma, default 0.5 mag. To reproduce NMMA's
# fixed --em-error-budget exactly, pin it instead with a line in a --priors
# file:  sigma = DeltaFunction(peak=0.25, name='sigma')
#
# When a budget runs it appears in the figure: the wider faint bar behind each
# point, with the residual strip and the chi-square divided by that same sigma.
# On SN2021ugl, quadrature measures sigma = 0.073 mag against photometric
# errors already at 0.113, so the budget is real but small. The default keeps
# the limits, which are 23 of 65 points and bound t0 from above.
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
  --likelihood upper-limits \
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
  --likelihood upper-limits \
  --nlive 2048 \
  --sampler pymultinest \
  --seed 42 \
  --plot-theme print \
  --sample-av-host \
  --av-host-max 1.0 \
  --ra 246.9820265 \
  --dec 20.253166509091 \
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
  --likelihood upper-limits \
  --nlive 2048 \
  --sampler pymultinest \
  --seed 42 \
  --plot-theme print \
  --sample-av-host \
  --av-host-max 1.0 \
  --ra 246.9820265 \
  --dec 20.253166509091 \
  --outdir results/redback


# The same template with a systematic error budget instead of the limits, the
# closest redback gets to the NMMA command at the end of this file. sigma is
# sampled, not given: --sigma-sys-max only bounds it. Its ln Z is not
# comparable with the three runs above, which keep the 23 limits.
.venv-redback/bin/python src/fitting/redback-fit/fit_redback.py \
  --object SN2021ugl \
  --data data/processed/SN2021ugl.dat \
  --model sncosmo_models \
  --sncosmo-model v19-1993j-corr \
  --priors priors_redback/sncosmo_generic.prior \
  --t0-min -30.0 \
  --t0-max 30.0 \
  --likelihood quadrature \
  --detections-only \
  --sigma-sys-max 0.5 \
  --nlive 2048 \
  --sampler pymultinest \
  --seed 42 \
  --plot-theme print \
  --sample-av-host \
  --av-host-max 1.0 \
  --ra 246.9820265 \
  --dec 20.253166509091 \
  --outdir results/redback

# ---------------------------------------------------------------------------
# NMMA, same template and same data, for comparison. Not part of the redback
# pipeline; left here so the two can be run back to back.
#
# Three things this command needs that are easy to miss:
#   --em-tmin -20   NMMA checks that the model can start before the first data
#                   point in the worst case of the priors. With timeshift up to
#                   +30 and z up to 1.025, -10 lets it start at +9.75 d, after
#                   the first point at 3.99 d, and it refuses.
#   --em-tmax  80   the same check at the other end: with timeshift down to -30,
#                   57 would end at +27 d, before the last point at 46.92 d.
#   priors_nmma/    a local copy of sncosmo-generic.prior with
#                   luminosity_distance minimum raised off 0, else
#                   z_at_value raises CosmologyError.
#
# Its ln Z is NOT comparable with redback's: NMMA reports noise evidence 0
# where redback reports -879948, so the two are normalised differently, and
# --em-error-budget 0.25 widens every error bar here and not there.
# ---------------------------------------------------------------------------
.venv-nmma/bin/lightcurve-analysis \
  --em-model v19-1993j-corr \
  --light-curve-data data/processed/SN2021ugl.dat \
  --prior-file priors_nmma/sncosmo-generic.prior \
  --outdir /tmp/nv --label v19_SN2021ugl \
  --trigger-time 59419.226053 \
  --em-tmin -20.0 \
  --em-tmax 80.0 \
  --data-tmin 0.0 \
  --data-tmax 47.0 \
  --em-tstep 0.25 \
  --em-error-budget 0.25 \
  --nlive 2048 \
  --sampler pymultinest \
  --generation-seed 42 \
  --use-Ebv --Ebv-max 0.0637
