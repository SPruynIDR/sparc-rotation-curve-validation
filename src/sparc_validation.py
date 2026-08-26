"""
sparc_validation.py — reproduce the headline numbers of the Paper II
rotation-curve validation from raw, public SPARC data.

Pipeline
--------
1. Load Table1 (galaxy sample) and Table2 (per-point mass models).
2. Apply the documented exclusion: drop any galaxy with R_HI <= 0 in
   Table1 (undefined H I radius -> the model's x=r/R_HI is undefined).
   This removes 4 of the 175 SPARC galaxies (D512-2, D564-8, D631-7,
   NGC5907), leaving N=171 galaxies. See results/excluded_points.csv.
3. Compute the model prediction for every remaining point using the
   frozen constants in src/model.py (NOT re-fit here — see
   fit_constants.py to reproduce the original global optimization).
4. Report RMS, R^2 against V_obs, and compare against a baryons-only
   (no boundary term) baseline.
5. Run the permutation test (10,000 shuffles) against the null
   hypothesis that the galaxy-to-galaxy pairing carries no information.

Run:
    python src/download_data.py        # once, to fetch the raw tables
    python src/sparc_validation.py
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from model import v_pred_squared, v_bar_squared, baryonic_mass  # noqa: E402
from parse_mrt import read_mrt, column  # noqa: E402

DATA_DIR = Path(__file__).parent.parent / "data"
PERMUTATION_SEED = 20260710
N_PERMUTATIONS = 10_000

TABLE1_CANDIDATES = {
    "name": ["Galaxy", "Name"],
    "T": ["T"],
    "L36": ["L36", "L[3.6]", "L3.6"],
    "MHI": ["MHI", "M_HI", "logMHI"],
    "RHI": ["RHI", "R_HI"],
    "Q": ["Q"],
}
TABLE2_CANDIDATES = {
    "name": ["Galaxy", "Name"],
    "r": ["R", "r", "Rad"],
    "Vobs": ["Vobs", "V_obs"],
    "e_Vobs": ["e_Vobs"],
    "Vgas": ["Vgas", "V_gas"],
    "Vdisk": ["Vdisk", "V_disk"],
    "Vbul": ["Vbul", "V_bul"],
}


def _resolve(table, candidates: dict[str, list[str]]) -> dict[str, str]:
    colnames = list(getattr(table, "colnames", table.keys() if isinstance(table, dict) else []))
    resolved = {}
    missing = []
    for key, options in candidates.items():
        hit = next((c for c in options if c in colnames), None)
        if hit is None:
            hit = next(
                (c for c in colnames if c.lower() in [o.lower() for o in options]), None
            )
        if hit is None:
            missing.append(key)
        else:
            resolved[key] = hit
    if missing:
        raise KeyError(
            f"Could not find column(s) {missing} among {colnames}. "
            f"Edit the *_CANDIDATES dict at the top of sparc_validation.py "
            f"to match the actual downloaded file's headers."
        )
    return resolved


def load_data(data_dir: Path = DATA_DIR):
    t1_path = data_dir / "SPARC_Lelli2016c.mrt"
    t2_path = data_dir / "MassModels_Lelli2016c.mrt"
    if not t1_path.exists() or not t2_path.exists():
        raise FileNotFoundError(
            f"Expected {t1_path.name} and {t2_path.name} in {data_dir}. "
            f"Run `python src/download_data.py` first."
        )

    t1 =
