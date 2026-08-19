"""Submit one HTCondor job per object listed in a targets CSV.

One job per object, no MPI. The venv path is a parameter: each backend
runs in its own interpreter (.venv-redback, .venv-mosfit, .venv-nmma).

This script only knows HTCondor mechanics (writing wrapper scripts,
building submit descriptions, calling condor_submit). It does not know
any backend's specific CLI flags or how to compute them (e.g. nmma's
--ebv-max/--z-known): each backend owns a condor_args.py in its own
directory (see src/fitting/nmma-fit/condor_args.py) that prints the
right fit-script arguments for one object, one per line on stdout. This
script just calls that.

Each job gets its own standalone wrapper script (condor/jobs/{object}.sh)
with the full command baked in, not passed via the Condor "arguments"
line. That keeps every submitted job re-runnable by hand for debugging
(just execute the .sh directly) without reconstructing the command from
Condor's argument-quoting, and gives each job a durable, inspectable
record on disk independent of the Condor queue/log.

CSV columns: object, ra, dec, z_known (z_known optional, leave blank if
unknown, in which case luminosity_distance is left free in the fit). A
single object is just a one-row CSV.

Everything after -- is backend-specific and forwarded as-is to
<backend>/condor_args.py; this script never names those flags itself.

Usage:
    python condor/submit_batch.py --backend nmma --targets config/targets.csv \\
        --data-dir data/processed --outdir results/ \\
        -- --model v19-1993j-corr --prior config/priors/sncosmo-generic.prior
"""

import argparse
import csv
import logging
import shlex
import subprocess
import textwrap
from pathlib import Path

logger = logging.getLogger(__name__)

BACKEND_VENV = {
    "redback": ".venv-redback",
    "mosfit": ".venv-mosfit",
    "nmma": ".venv-nmma",
}
BACKEND_SCRIPT = {
    "redback": "src/fitting/redback-fit/fit_nickel.py",
    "mosfit": "src/fitting/mosfit-fit/fit_mosfit.py",
    "nmma": "src/fitting/nmma-fit/fit_nmma.py",
}
# Each backend's own arg-building script (see its docstring: prints the
# fit-script's CLI args, one per line, for one object).
BACKEND_CONDOR_ARGS_SCRIPT = {
    "nmma": "src/fitting/nmma-fit/condor_args.py",
    # redback/mosfit: add once fit_nickel.py/fit_mosfit.py are past the
    # NotImplementedError stub and their own condor_args.py exists.
}


def write_wrapper(
    venv: str, script: str, fit_args: list[str], jobs_dir: Path, label: str
) -> Path:
    """Write a standalone, directly re-runnable job script for this run."""
    jobs_dir.mkdir(parents=True, exist_ok=True)
    wrapper_path = jobs_dir / f"{label}.sh"
    cmd = shlex.join([f"{venv}/bin/python", script, *fit_args])
    wrapper_path.write_text(
        textwrap.dedent(f"""\
        #!/bin/bash
        set -euo pipefail
        export OMP_NUM_THREADS=1
        {cmd}
        """)
    )
    wrapper_path.chmod(0o755)
    return wrapper_path


def build_submit_description(
    wrapper_path: Path, log_dir: Path, request_memory_mb: int, request_disk_mb: int
) -> str:
    return textwrap.dedent(f"""\
        universe = vanilla
        executable = {wrapper_path}
        output = {log_dir}/$(Cluster)_$(Process).out
        error = {log_dir}/$(Cluster)_$(Process).err
        log = {log_dir}/$(Cluster)_$(Process).log
        request_cpus = 1
        request_memory = {request_memory_mb} MB
        request_disk = {request_disk_mb} MB
        on_exit_remove = (ExitBySignal == False) && (ExitCode == 0)
        on_exit_hold = (ExitBySignal == True) || (ExitCode != 0)
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
    """Ask <backend>/condor_args.py to build the fit-script args for one object."""
    condor_args_script = BACKEND_CONDOR_ARGS_SCRIPT.get(backend)
    if condor_args_script is None:
        raise NotImplementedError(
            f"--backend {backend}: no condor_args.py wired up yet"
        )

    cmd = [
        f"{venv}/bin/python",
        condor_args_script,
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
            f"{condor_args_script} failed for {target['object']}: {proc.stderr.strip()}"
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
    parser.add_argument("--request-memory-mb", type=int, default=8192)
    parser.add_argument("--request-disk-mb", type=int, default=500)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the per-object wrapper scripts but skip condor_submit "
        "(useful to generate/test the .sh files without Condor installed)",
    )
    parser.add_argument(
        "backend_args",
        nargs=argparse.REMAINDER,
        help="Everything after -- is forwarded as-is to <backend>/condor_args.py",
    )
    args = parser.parse_args()
    if args.backend_args and args.backend_args[0] == "--":
        args.backend_args = args.backend_args[1:]
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    venv = BACKEND_VENV[args.backend]
    for target in read_targets(args.targets):
        fit_args = get_fit_args(args.backend, venv, target, args)
        wrapper_path = write_wrapper(
            venv,
            BACKEND_SCRIPT[args.backend],
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
        submit_description = build_submit_description(
            wrapper_path, log_dir, args.request_memory_mb, args.request_disk_mb
        )
        submit(submit_description)
        logger.info("Submitted %s (%s)", target["object"], args.backend)


if __name__ == "__main__":
    main()
