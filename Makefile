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

.PHONY: setup-redback setup-mosfit setup-nmma setup-multinest setup-all setup-precommit submit-nmma

#  ============ Redback ============
setup-redback:
	uv venv .venv-redback
	uv pip install --python .venv-redback -r requirements-redback.txt

#  ============ Mosfit ============
setup-mosfit:
	uv venv .venv-mosfit
	uv pip install --python .venv-mosfit -r requirements-mosfit.txt


#  ============ NMMA ============

# nmma's lightcurve-analysis defaults to --sampler dynesty in this repo (no
# native build needed); pymultinest is still built and installed so
# --sampler pymultinest remains available on request.
#
# nmma is installed from GitHub, not PyPI (the 1.0.1 release has known bugs
# already fixed upstream), pinned to a specific commit so `make setup-nmma`
# stays reproducible instead of silently tracking main.
NMMA_COMMIT := 4342b6abb1d70eb6f3e60afcc138662386d3c29a
setup-nmma: setup-multinest
	uv venv .venv-nmma
	uv pip install --python .venv-nmma -r requirements-nmma.txt
	rm -rf /tmp/nmma
	git clone https://github.com/nuclear-multimessenger-astronomy/nmma /tmp/nmma
	cd /tmp/nmma && git checkout $(NMMA_COMMIT)
	uv pip install --python .venv-nmma /tmp/nmma
	rm -rf /tmp/pymultinest
	git clone https://github.com/JohannesBuchner/PyMultiNest /tmp/pymultinest
	uv pip install --python .venv-nmma /tmp/pymultinest
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
