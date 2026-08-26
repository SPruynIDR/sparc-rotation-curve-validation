"""
SPARC Rotation Curve Validation Pipeline
----------------------------------------
Loads SPARC Table 1 (galaxy properties) and Table 2 (rotation curve points),
applies the global boundary scaling relation, filters invalid/unbounded targets,
and generates validation summary diagnostics.
"""

from pathlib import Path
import numpy as np
from parse_mrt import read_mrt, column

TABLE1_CANDIDATES = {
    "Name": ["Galaxy", "Name"],
    "RHI": ["RHI", "R_HI"],
    "L36": ["L36", "L_36"],
    "MHI": ["MHI", "M_HI"],
    "T": ["T", "Type"]
}

TABLE2_CANDIDATES = {
    "Name": ["Galaxy", "Name", "ID"],
    "r": ["r", "R"],
    "Vobs": ["Vobs"],
    "Vgas": ["Vgas"],
    "Vdisk": ["Vdisk"],
    "Vbul": ["Vbul"]
}

def _resolve(table, mapping):
    if hasattr(table, "colnames"):
        available = list(table.colnames)
    else:
        available = list(table.keys())
    
    resolved = {}
    for key, candidates in mapping.items():
        matched = False
        for cand in candidates:
            if cand in available:
                resolved[key] = cand
                matched = True
                break
        if not matched:
            raise KeyError(f"Could not find valid column for {key} among {candidates} in {available}")
    return resolved

def load_data(data_dir: Path = Path("data")):
    # Check data/sparc first, fallback to data/
    sparc_subdir = data_dir / "sparc"
    if (sparc_subdir / "Table1.mrt").exists():
        data_dir = sparc_subdir

    t1_path = data_dir / "Table1.mrt"
    t2_path = data_dir / "MassModels_Lelli2016c.mrt"
    
    if not t1_path.exists() or not t2_path.exists():
        raise FileNotFoundError(
            f"Expected {t1_path.name} and {t2_path.name} in {data_dir}. "
            "Run 'python src/download_data.py' first."
        )

    t1 = read_mrt(t1_path)
    t2 = read_mrt(t2_path)

    c1 = _resolve(t1, TABLE1_CANDIDATES)
    c2 = _resolve(t2, TABLE2_CANDIDATES)

    names1 = np.asarray(column(t1, c1["Name"]), dtype=str)
    RHI = column(t1, c1["RHI"])
    L36 = column(t1, c1["L36"])
    MHI = column(t1, c1["MHI"])
    T = column(t1, c1["T"])

    good = RHI > 0
    excluded_names = set(names1[~good])
    print(f"[info] excluding {len(excluded_names)} galaxies with R_HI<=0: {list(excluded_names)}")

    galaxy_table = {
        name: {"RHI": rhi, "L36": l36, "MHI": mhi, "T": t}
        for name, rhi, l36, mhi, t, keep in zip(names1, RHI, L36, MHI, T, good)
        if keep
    }

    names2 = np.asarray(column(t2, c2["Name"]), dtype=str)
    r = column(t2, c2["r"])
    Vobs = column(t2, c2["Vobs"])
    Vgas = column(t2, c2["Vgas"])
    Vdisk = column(t2, c2["Vdisk"])
    Vbul = column(t2, c2["Vbul"])

    keep_point = np.array([n in galaxy_table for n in names2])
    print(
        f"[info] {keep_point.sum()} of {len(names2)} points retained after "
        f"the galaxy-level exclusion (target: 3346 of ~3350 across 171 galaxies)"
    )

    rows = []
    for i in np.where(keep_point)[0]:
        g = galaxy_table[names2[i]]
        rows.append(
            dict(
                name=names2[i],
                r=r[i],
                Vobs=Vobs[i],
                Vgas=Vgas[i],
                Vdisk=Vdisk[i],
                Vbul=Vbul[i],
                RHI=g["RHI"],
                L36=g["L36"],
                MHI=g["MHI"],
                T=g["T"],
            )
        )

    return rows, list(galaxy_table.keys()), excluded_names

def main():
    data, included_galaxies, excluded_galaxies = load_data()
    print(f"[info] Successfully processed {len(included_galaxies)} galaxies and {len(data)} data points.")

if __name__ == "__main__":
    main()
    
