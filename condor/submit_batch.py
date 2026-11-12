"""Submit one HTCondor job per object listed in a targets CSV.

One job per object, no MPI. The venv path is a parameter: each backend
runs in its own interpreter (.venv-redback, .venv-mosfit, .venv-nmma).

This script only knows HTCondor mechanics (writing wrapper scripts,
building submit descriptions, calling condor_submit). It does not know
any backend's specific CLI flags or how to compute them (e.g. nmma's
--ebv-max/--z-known): each backend owns a _condor module in its own
subpackage (see carich.fitting.nmma._condor) that prints the right
fit arguments for one object, one per line on stdout. This script just
calls that.

Each job gets its own standalone wrapper script (condor/jobs/{object}.sh)
with the full command baked in, not passed via the Condor "arguments"
line. That keeps every submitted job re-runnable by hand for debugging
(just execute the .sh directly) without reconstructing the command from
Condor's argument-quoting, and gives each job a durable, inspectable
record on disk independent of the Condor queue/log.

The memory and disk requests are ClassAd expressions rather than constants:
a job held for exceeding either limit is released and retried with a larger
request, up to a ceiling. Starting low keeps the job matchable on many slots.
The LIGO pools refuse a job without an accounting group, so
--accounting-group is required for a real submission.

CSV columns: object, ra, dec, z_known (z_known optional, leave blank if
unknown, in which case luminosity_distance is left free in the fit). A
single object is just a one-row CSV.

Everything after -- is backend-specific and forwarded as-is to
the _condor module of the backend; this script never names those flags
itself.

Usage:
    python condor/submit_batch.py --backend nmma --targets config/targets.csv \\
        --data-dir data/processed --outdir results/ \\
        --accounting-group <tag> \\
        -- --model v19-1993j-corr --prior config/priors/sncosmo-generic.prior
"""

import argparse
import csv
import logging
import shlex
import subprocess
import sys
import textwrap
from pathlib import Path

logger = logging.getLogger(__name__)

BACKEND_VENV = {
    "redback": ".venv-redback",
    "mosfit": ".venv-mosfit",
    "nmma": ".venv-nmma",
}
# Run with `python -m`, from the environment where carich is installed.
BACKEND_MODULE = {
    "redback": "carich.fitting.redback._fit",
    "mosfit": "carich.fitting.mosfit._fit",
    "nmma": "carich.fitting.nmma._fit",
}
# Each backend's own arg-building module (see its docstring: prints the
# CLI args of the fit, one per line, for one object).
BACKEND_CONDOR_ARGS_MODULE = {
    "nmma": "carich.fitting.nmma._condor",
    "redback": "carich.fitting.redback._condor",
    # mosfit: add once its _fit is past the NotImplementedError stub and
    # has its own _condor.
}


# Hold reason codes that an escalating retry can fix: 34 is the memory limit,
# 104 the disk limit.
HOLD_MEMORY = 34
HOLD_DISK = 104

# pymultinest loads libmultinest when it is imported, and the wrapper calls
# the interpreter of the venv without its activate script, which is what
# exports this. `make setup-multinest` installs the library there.
MULTINEST_DIR = Path.home() / ".local" / "lib"
LIBRARY_PATH_VARIABLE = (
    "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
)


def write_wrapper(
    venv: str, module: str, fit_args: list[str], jobs_dir: Path, label: str
) -> Path:
    """Write a standalone, directly re-runnable job script for this run."""
    jobs_dir.mkdir(parents=True, exist_ok=True)
    wrapper_path = jobs_dir / f"{label}.sh"
    cmd = shlex.join([f"{venv}/bin/python", "-m", module, *fit_args])
    wrapper_path.write_text(
        textwrap.dedent(f"""\
        #!/bin/bash
        set -euo pipefail
        export OMP_NUM_THREADS=1
        export {LIBRARY_PATH_VARIABLE}="{MULTINEST_DIR}:${{{LIBRARY_PATH_VARIABLE}:-}}"
        cd {shlex.quote(str(Path.cwd()))}
        {cmd}
        """)
    )
    wrapper_path.chmod(0o755)
    return wrapper_path


def escalate(usage: str, floor: int, cap: int, growth: float) -> str:
    """ClassAd request expression of one resource.

    A retry asks for ``growth`` times what the previous attempt consumed,
    never below the floor and never above the cap. ``usage`` (``MemoryUsage``
    or ``DiskUsage``) does not exist until the job has run, so the expression
    falls back to the floor at submit time; otherwise it evaluates to
    undefined and the schedd refuses the job.
    """
    return (
        f"ifthenelse({usage} =!= undefined, "
        f"int(min({{{cap}, max({{{floor}, {usage} * {growth}}})}})), {floor})"
    )


def build_submit_description(
    wrapper_path: Path, log_dir: Path, args: argparse.Namespace
) -> str:
    """Submit description of one fit, ready for ``condor_submit``.

    The memory ceiling must stay below the RAM of the largest slot of the
    pool (``condor_status -af Memory | sort -n | tail -1``); above it, an
    escalating job stops matching anything and sits idle instead of failing.
    """
    # request_memory is expressed in MiB, request_disk in KiB.
    memory = args.request_memory_mb
    memory_cap = max(memory, args.max_request_memory_mb)
    disk = args.request_disk_mb * 1024
    disk_cap = max(disk, args.max_request_disk_mb * 1024)
    memory_request = escalate("MemoryUsage", memory, memory_cap, args.growth_factor)
    disk_request = escalate("DiskUsage", disk, disk_cap, args.growth_factor)
    releasable = f"(HoldReasonCode == {HOLD_MEMORY} || HoldReasonCode == {HOLD_DISK})"
    return textwrap.dedent(f"""\
        +MaxHours = {args.max_hours}
        universe = vanilla
        accounting_group = {args.accounting_group}
        executable = {wrapper_path}
        output = {log_dir}/$(Cluster)_$(Process).out
        error = {log_dir}/$(Cluster)_$(Process).err
        log = {log_dir}/$(Cluster)_$(Process).log
        request_cpus = 1
        request_memory = {memory_request}
        request_disk = {disk_request}
        periodic_release = {releasable} && (NumJobStarts <= {args.max_retries})
        on_exit_remove = (ExitBySignal == False) && (ExitCode == 0)
        on_exit_hold = (ExitBySignal == True) || (ExitCode != 0)
        on_exit_hold_subcode = ExitCode
        on_exit_hold_reason = (ExitBySignal == True \\
            ? strcat("job exited with signal ", ExitSignal) \\
            : strcat("job exited with code ", ExitCode))
        getenv = false
        queue 1
        """)


def submit(submit_description: str) -> None:
    proc = subprocess.run(
        ["condor_submit"], input=submit_description, text=True, capture_output=True
    )
    if proc.returncode != 0:
        raise RuntimeError(f"condor_submit failed: {proc.stderr.strip()}")
    logger.info(proc.stdout.strip())


def read_targets(path: Path) -> list[dict]:
    with open(path) as fh:
        return list(csv.DictReader(fh))


def get_fit_args(
    backend: str, venv: str, target: dict, args: argparse.Namespace
) -> list[str]:
    """Ask the _condor module of the backend for the fit args of one object."""
    condor_args_module = BACKEND_CONDOR_ARGS_MODULE.get(backend)
    if condor_args_module is None:
        raise NotImplementedError(f"--backend {backend}: no _condor wired up yet")

    cmd = [
        f"{venv}/bin/python",
        "-m",
        condor_args_module,
        "--object",
        target["object"],
        "--ra",
        target["ra"],
        "--dec",
        target["dec"],
        "--data-dir",
        args.data_dir,
        "--outdir",
        args.outdir,
        *args.backend_args,
    ]
    if target.get("z_known"):
        cmd += ["--z-known", target["z_known"]]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{condor_args_module} failed for {target['object']}: {proc.stderr.strip()}"
        )
    return proc.stdout.strip("\n").split("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--backend", required=True, choices=sorted(BACKEND_VENV))
    parser.add_argument("--targets", required=True, type=Path)
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--outdir", default="results/")
    parser.add_argument("--jobs-dir", type=Path, default=Path("condor/jobs"))
    parser.add_argument(
        "--accounting-group",
        default=None,
        help="Accounting tag of the jobs, e.g. ligo.dev.o4.cbc.pe.bayestar. "
        "Required unless --dry-run",
    )
    parser.add_argument("--max-hours", type=int, default=24)
    parser.add_argument(
        "--request-memory-mb",
        type=int,
        default=8192,
        help="Memory of the first attempt; a retry asks for more",
    )
    parser.add_argument("--request-disk-mb", type=int, default=500)
    parser.add_argument("--max-request-memory-mb", type=int, default=32768)
    parser.add_argument("--max-request-disk-mb", type=int, default=4000)
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Releases of a job held for exceeding its memory or disk request",
    )
    parser.add_argument(
        "--growth-factor",
        type=float,
        default=2.0,
        help="A retry requests this many times the measured usage",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the per-object wrapper scripts but skip condor_submit "
        "(useful to generate/test the .sh files without Condor installed)",
    )
    parser.add_argument(
        "backend_args",
        nargs=argparse.REMAINDER,
        help="Everything after -- is forwarded as-is to the backend's _condor",
    )
    args = parser.parse_args()
    if args.backend_args and args.backend_args[0] == "--":
        args.backend_args = args.backend_args[1:]
    if args.accounting_group is None and not args.dry_run:
        parser.error("--accounting-group is required unless --dry-run is given")
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    venv = BACKEND_VENV[args.backend]
    for target in read_targets(args.targets):
        fit_args = get_fit_args(args.backend, venv, target, args)
        wrapper_path = write_wrapper(
            venv,
            BACKEND_MODULE[args.backend],
            fit_args,
            args.jobs_dir,
            target["object"],
        )

        if args.dry_run:
            logger.info("[dry-run] Wrote %s, not submitting", wrapper_path)
            continue

        # Condor's own job logs (.out/.err/.log) live alongside the fit's
        # own results, not in a separate shared directory: everything
        # for one object stays in one place.
        log_dir = Path(args.outdir) / target["object"]
        log_dir.mkdir(parents=True, exist_ok=True)
        submit_description = build_submit_description(wrapper_path, log_dir, args)
        submit(submit_description)
        logger.info("Submitted %s (%s)", target["object"], args.backend)


if __name__ == "__main__":
    main()
