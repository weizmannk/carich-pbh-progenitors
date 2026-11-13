"""Reduce the classified sample to candidates and controls."""

import argparse
from pathlib import Path

import pandas as pd

from ..selection import read_crossmatch, select


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("classified", type=Path, help="tab-separated classifications")
    p.add_argument("crossmatch", type=Path, help="CSV of transient-galaxy pairs")
    p.add_argument("lightcurves", type=Path, help="directory of lc_<name>.csv")
    p.add_argument("--outdir", type=Path, default=Path("data/selection"))
    p.add_argument("--max-dz", type=float, default=0.005)
    p.add_argument("--faint-limit", type=float, default=-17.5)
    p.add_argument(
        "--no-extinction",
        action="store_true",
        help="skip the Milky Way correction (no dust map download)",
    )
    return p


def main(args=None) -> None:
    opts = parser().parse_args(args)
    classified = pd.read_csv(opts.classified, sep="\t").drop_duplicates("source_name")

    Ebv = None
    if not opts.no_extinction:
        from ..ingest._extinction import get_planck_ebv

        Ebv = [
            get_planck_ebv(ra, dec) for ra, dec in zip(classified.ra, classified.dec)
        ]

    table = select(
        classified,
        read_crossmatch(opts.crossmatch),
        opts.lightcurves,
        Ebv=Ebv,
        max_dz=opts.max_dz,
        faint_limit=opts.faint_limit,
    )

    opts.outdir.mkdir(parents=True, exist_ok=True)
    table.to_csv(opts.outdir / "selection.csv", index=False, float_format="%.5g")
    for status in ("candidate", "control"):
        subset = table[table.status == status].sort_values("peak_absolute_magnitude")
        subset.to_csv(opts.outdir / f"{status}s.csv", index=False, float_format="%.5g")
    # Same columns as config/targets.csv, read by condor/submit_batch.py.
    targets = table.loc[table.status == "candidate", ["name", "ra", "dec", "redshift"]]
    targets.columns = ["object", "ra", "dec", "z_known"]
    targets.to_csv(opts.outdir / "targets.csv", index=False)

    counts = table.groupby(["status", "reason"]).size().sort_values(ascending=False)
    print(counts.to_string())
    copied = table.loc[table.status != "rejected", "redshift_copied"].sum()
    print(f"\n{copied} retained objects share the redshift of their host exactly;")
    print("measure it in the spectrum of the transient before trusting the match.")


if __name__ == "__main__":
    main()
