# ==============================================================================
# carich-pbh-progenitors - environment setup
# ==============================================================================
#
# Usage:
#   make setup-redback  : .venv-redback  (redback, bilby, dynesty, dustmaps)
#   make setup-mosfit   : .venv-mosfit   (MOSFiT, needs mpi4py -> brew install open-mpi)
#   make setup-nmma     : .venv-nmma     (nmma; also builds MultiNest + PyMultiNest)
#   make setup-all      : all three
#   make setup-precommit: install pre-commit and its git hook (once, dev-only)
#   make submit-nmma    : submit condor/submit_batch.py --backend nmma
#                          (override TARGETS/MODEL/PRIOR, e.g. make submit-nmma MODEL=nugent-sn1a)
#   make submit-redback : same for redback (override REDBACK_MODEL)
#   make test           : redback pipeline tests, without the sampling ones
#   make test-all       : including the injection-recovery test (slow)
#
# Prerequisites (Mac):
#   brew install cmake gfortran openblas open-mpi
# ==============================================================================

UNAME := $(shell uname -s)

ifeq ($(UNAME), Darwin)
    BLAS_LIB := /opt/homebrew/opt/openblas/lib/libopenblas.dylib
    LAPACK_LIB := /opt/homebrew/opt/openblas/lib/libopenblas.dylib
    LIB_EXT := dylib
    LIB_PATH_VAR := DYLD_LIBRARY_PATH
else
    ARCH := $(shell dpkg-architecture -qDEB_HOST_MULTIARCH 2>/dev/null || echo x86_64-linux-gnu)
    BLAS_LIB := /usr/lib/$(ARCH)/libopenblas.so
    LAPACK_LIB := /usr/lib/$(ARCH)/liblapack.so
    LIB_EXT := so
    LIB_PATH_VAR := LD_LIBRARY_PATH
endif

.PHONY: setup-redback setup-mosfit setup-nmma setup-multinest setup-all \
	setup-precommit submit-nmma submit-redback test test-all

#  ============ Mosfit ============
# Python 3.12 is the only version that satisfies both the package
# (>= 3.12) and the numpy <= 1.26.4 that MOSFiT pins.
setup-mosfit:
	uv venv .venv-mosfit --python 3.12
	uv pip install --python .venv-mosfit -e ".[mosfit]"


#  ============ NMMA ============

# Default sampler is dynesty; pymultinest is still installed for --sampler
# pymultinest on request.
#
# nmma comes from GitHub; its version is pinned in the nmma extra of
# pyproject.toml.
# nmma requires Python >=3.12,<3.14; pinned explicitly because `uv venv` with
# no --python picks whatever the machine defaults to, and 3.11 fails to resolve.
setup-nmma: setup-multinest
	uv venv .venv-nmma --python 3.12
	uv pip install --python .venv-nmma -e ".[nmma]"
	uv pip install --python .venv-nmma "git+https://github.com/JohannesBuchner/PyMultiNest"
	echo 'export $(LIB_PATH_VAR)=$(HOME)/.local/lib:$$$(LIB_PATH_VAR)' >> .venv-nmma/bin/activate

setup-multinest:
	rm -rf /tmp/multinest
	git clone https://github.com/JohannesBuchner/MultiNest /tmp/multinest
	cd /tmp/multinest/build && cmake .. \
		-DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_DISABLE_FIND_PACKAGE_MPI=ON \
		-DMPI_C_COMPILER="" \
		-DMPI_CXX_COMPILER="" \
		-DMPI_Fortran_COMPILER="" \
		-DBLAS_LIBRARIES=$(BLAS_LIB) \
		-DLAPACK_LIBRARIES=$(LAPACK_LIB)
	cd /tmp/multinest/build && make -j1
	mkdir -p $(HOME)/.local/lib
	cp /tmp/multinest/lib/libmultinest.$(LIB_EXT) $(HOME)/.local/lib/
	cp /tmp/multinest/lib/libmultinest.a $(HOME)/.local/lib/


setup-all: setup-redback setup-mosfit setup-nmma

#  ============ Tests ============
# They import carich.fitting.redback, so they run in .venv-redback.
REDBACK_TESTS := src/carich/fitting/redback

test:
	.venv-redback/bin/python -m pytest $(REDBACK_TESTS) -m "not slow" -q

test-all:
	.venv-redback/bin/python -m pytest $(REDBACK_TESTS) -q

#  ============ Dev tooling ============
# Independent of the three fitting venvs: pre-commit runs at commit time,
# not inside any of redback/mosfit/nmma.
setup-precommit:
	uv tool install pre-commit
	pre-commit install

#  ============ Submission ============
TARGETS ?= config/targets.csv
MODEL ?= v19-1993j-corr
PRIOR ?= config/priors/sncosmo-generic.prior

submit-nmma:
	.venv-nmma/bin/python condor/submit_batch.py --backend nmma --targets $(TARGETS) \
		--data-dir data/processed --outdir results/ \
		-- --model $(MODEL) --prior $(PRIOR)

# redback takes its own model names (see carich.fitting.redback.MODELS) and finds its
# prior in priors_redback/<model>.prior, so neither MODEL nor PRIOR above
# applies. Override REDBACK_MODEL, e.g. make submit-redback REDBACK_MODEL=slsn.
REDBACK_MODEL ?= arnett

submit-redback:
	.venv-redback/bin/python condor/submit_batch.py --backend redback --targets $(TARGETS) \
		--data-dir data/processed --outdir results/ \
		-- --model $(REDBACK_MODEL)

# ------------------------------------------------------------------------
# redback: its own environment.
#
# Each fitter gets its own venv. With nmma v1.0.2 the two are closer than they
# used to be, both resolve bilby 2.8.2, but they still disagree on the
# array stack (measured: redback numpy 2.5.4 / pandas 3.0.6, nmma numpy 2.3.5 /
# pandas 3.1.0), and MOSFiT pins numpy <=1.26.4, which neither tolerates.
# Separate interpreters also keep a redback upgrade from silently changing an
# NMMA result, which matters more here than the disk space.
#
# pymultinest is the default sampler and loads libmultinest at import time,
# so setup-multinest must have run first.
# ------------------------------------------------------------------------
setup-redback: setup-multinest
	uv venv .venv-redback --python 3.12
	uv pip install --python .venv-redback -e ".[redback]"
	echo 'export $(LIB_PATH_VAR)=$(HOME)/.local/lib:$$$(LIB_PATH_VAR)' \
		>> .venv-redback/bin/activate
