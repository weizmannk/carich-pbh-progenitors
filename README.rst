carich-pbh-progenitors
======================

Faint, hydrogen-poor transients as a test of white dwarf explosions triggered
by dark matter.

A primordial black hole crossing a white dwarf can ignite it well below the
Chandrasekhar mass. The explosion is then the detonation of a bare white
dwarf: little nickel, a fast and faint light curve, no hydrogen. This
repository selects such transients among the spectroscopically classified
ZTF sample, gathers their photometry, and fits their light curves with
redback, NMMA and MOSFiT to measure the nickel and ejecta masses that the
detonation models predict.

Installation
------------

The three fitters disagree on numpy and pandas, so each one gets its own
environment, built with `uv <https://docs.astral.sh/uv/>`_ from an extra of
``pyproject.toml``. The package is installed in editable mode in each of
them, because the data and the prior files stay in the repository::

    brew install cmake gfortran openblas open-mpi   # macOS
    make setup-redback    # .venv-redback, the reference fits and the tests
    make setup-nmma       # .venv-nmma, template fits for comparison
    make setup-mosfit     # .venv-mosfit, cross-check, not implemented yet
    make test             # redback pipeline tests, without the sampling ones

Never install two extras in the same environment. ``make setup-redback``
and ``make setup-nmma`` also build MultiNest, which pymultinest loads when it
is imported.

Commands
--------

``carich-select`` reduces the classified sample to candidates and controls.
It keeps the hydrogen-poor supernovae whose redshift agrees with a galaxy of
the CLU catalog, and separates them at a peak absolute magnitude of -17.5::

    carich-select data/catalogs/ZTF_Gap_Trasient_classifications.dat \
        data/catalogs/crossmatch_clu_dist20.0-200.0Mpc_projdist10-130.csv \
        <directory of lc_<name>.csv alert light curves>

It writes ``selection.csv``, ``candidates.csv``, ``controls.csv`` and
``targets.csv`` in ``data/selection``. The Milky Way extinction comes from
the Planck GNILC map, downloaded on first use; ``--no-extinction`` skips it.

``carich-fetch-atlas`` adds ATLAS forced photometry to the light curve of one
object, stacked by night, in ``data/processed/<name>.dat``. It needs the
token of a free account on https://fallingstar-data.com/forcedphot/ in the
environment variable ``ATLAS_TOKEN``::

    carich-fetch-atlas SN2021ugl 246.9820265 20.253166509091 59400 59480

The fits are run as modules, one object and one model at a time;
``run_redback_SN2021ugl.sh`` holds worked examples, and
``condor/submit_batch.py`` submits one HTCondor job per row of
``config/targets.csv``::

    .venv-redback/bin/python -m carich.fitting.redback._fit \
        --object SN2021ugl --model arnett --outdir results/redback

Layout
------

::

    carich-pbh-progenitors/
    ├── pyproject.toml          package "carich", extras redback/nmma/mosfit
    ├── Makefile                environments, tests, Condor submission
    ├── config/                 targets and NMMA priors of the batch runs
    ├── condor/                 HTCondor submission
    ├── priors_redback/         one prior file per redback model
    ├── priors_nmma/
    ├── data/
    │   ├── catalogs/           classifications and CLU crossmatch
    │   ├── models/             detonation yields of Sim et al. (2010)
    │   ├── processed/          photometry read by the fitters
    │   └── selection/          output of carich-select
    └── src/carich/
        ├── _paths.py           root of the repository, data and priors
        ├── _cli/               carich-select, carich-fetch-atlas
        ├── selection/          host association, peak luminosity, yields
        ├── ingest/             ATLAS, ZTF brokers, Astro-COLIBRI, dust maps
        └── fitting/
            ├── redback/        physical models, with their tests
            ├── nmma/           supernova templates
            └── mosfit/         placeholder
