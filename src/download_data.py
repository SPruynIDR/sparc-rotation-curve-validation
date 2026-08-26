"""
download_data.py — fetch the two official SPARC data tables this analysis
needs, directly from the SPARC team's own site, and cache them locally.

This repository does NOT bundle SPARC's data files. SPARC is maintained by
Lelli, McGaugh & Schombert and should be fetched from the canonical source
so users always get the current, correctly-attributed files. Run this once
before sparc_validation.py or model_comparison.py.

Source: http://astroweb.cwru.edu/SPARC/
Citation: Lelli, F., McGaugh, S. S., & Schombert, J. M. 2016, AJ, 152, 157

    Table1  -> SPARC_Lelli2016c.mrt   (galaxy sample: L_[3.6], M_HI, R_HI, T, Q, ...)
    Table2  -> MassModels_Lelli2016c.mrt  (per-point mass models: r, Vobs, Vgas, Vdisk, Vbul)

Usage:
    python src/download_data.py                # downloads into ./data/
    python src/download_data.py --data-dir X    # downloads into X/
"""

from __future__ import annotations
import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

BASE_URL = "https://astroweb.cwru.edu/SPARC/"
FILES = {
    "SPARC_Lelli2016c.mrt": "table1",       # galaxy sample table
    "MassModels_Lelli2016c.mrt": "table2",  # mass models table
}


def download(data_dir: Path, force: bool = False) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for filename in FILES:
        dest = data_dir / filename
        if dest.exists() and not force:
            print(f"[skip] {filename} already present at {dest}")
            continue
        url = BASE_URL + filename
        print(f"[fetch] {url}")
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception as exc:  # noqa: BLE001 - report and re-raise for CI visibility
            print(f"[error] could not download {url}: {exc}", file=sys.stderr)
            print(
                "If astroweb.cwru.edu is temporarily unreachable, the same "
                "tables are also mirrored via VizieR (CDS), DOI "
                "10.26093/cds/vizier.51520157 — see the paper's Data "
                "Availability Statement.",
                file=sys.stderr,
            )
            raise
        size = dest.stat().st_size
        sha256 = hashlib.sha256(dest.read_bytes()).hexdigest()
        print(f"[ok] {filename}: {size:,} bytes, sha256={sha256}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", default="data", help="directory to download into (default: ./data)"
    )
    parser.add_argument(
        "--force", action="store_true", help="re-download even if files already exist"
    )
    args = parser.parse_args()
    download(Path(args.data_dir), force=args.force)
    print("\nDone. Run src/sparc_validation.py next.")


if __name__ == "__main__":
    main()
