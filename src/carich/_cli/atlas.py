"""Add ATLAS forced photometry to the light curve of one object."""

import argparse
from pathlib import Path

from ..ingest import clean, read, request, stack, write


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("name", help="object name; the output is <outdir>/<name>.dat")
    p.add_argument("ra", type=float, help="degrees")
    p.add_argument("dec", type=float, help="degrees")
    p.add_argument("mjd_min", type=float)
    p.add_argument("mjd_max", type=float)
    p.add_argument("--outdir", type=Path, default=Path("data/processed"))
    return p


def main(args=None) -> None:
    opts = parser().parse_args(args)
    text = request(opts.ra, opts.dec, opts.mjd_min, opts.mjd_max)
    nights = stack(clean(read(text)))
    write(nights, opts.outdir / f"{opts.name}.dat", append=True)
    detections = int((nights.magerr < float("inf")).sum())
    print(f"{opts.name}: {len(nights)} nights, {detections} detections")


if __name__ == "__main__":
    main()
