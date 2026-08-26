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
    "name": ["ID", "Galaxy", "Name"],
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

    t1 = read_mrt(t1_path)
    t2 = read_mrt(t2_path)
    print(f"[info] Table1 columns: {getattr(t1, 'colnames', list(t1.keys()))}")
    print(f"[info] Table2 columns: {getattr(t2, 'colnames', list(t2.keys()))}")

    c1 = _resolve(t1, TABLE1_CANDIDATES)
    c2 = _resolve(t2, TABLE2_CANDIDATES)

    names1 = np.asarray(t1[c1["name"]], dtype=str)
    RHI = column(t1, c1["RHI"])
    L36 = column(t1, c1["L36"])
    MHI = column(t1, c1["MHI"])
    T = column(t1, c1["T"])

    good = RHI > 0
    excluded_names = names1[~good]
    print(f"[info] excluding {(~good).sum()} galaxies with R_HI<=0: {list(excluded_names)}")

    galaxy_table = {
        name: {"RHI": rhi, "L36": l36, "MHI": mhi, "T": t}
        for name, rhi, l36, mhi, t, keep in zip(names1, RHI, L36, MHI, T, good)
        if keep
    }

    names2 = np.asarray(t2[c2["name"]], dtype=str)
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
    return rows, galaxy_table, excluded_names


def compute_predictions(rows):
    r = np.array([row["r"] for row in rows])
    Vobs = np.array([row["Vobs"] for row in rows])
    Vgas = np.array([row["Vgas"] for row in rows])
    Vdisk = np.array([row["Vdisk"] for row in rows])
    Vbul = np.array([row["Vbul"] for row in rows])
    RHI = np.array([row["RHI"] for row in rows])
    L36 = np.array([row["L36"] for row in rows])
    MHI = np.array([row["MHI"] for row in rows])

    V_bar2 = v_bar_squared(Vgas, Vdisk, Vbul)
    M_bar = baryonic_mass(MHI, L36)
    x = r / RHI

    V_pred2 = v_pred_squared(V_bar2, M_bar, x, clip_negative_vbar2=True)
    V_pred = np.sqrt(V_pred2)
    V_bar2_clipped = np.clip(V_bar2, 0, None)
    V_bar = np.sqrt(V_bar2_clipped)

    return Vobs, V_pred, V_bar


def rms(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def r_squared(obs, pred):
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    return float(1 - ss_res / ss_tot)


def permutation_test(rows, Vobs, V_pred, seed=PERMUTATION_SEED, n_perm=N_PERMUTATIONS):
    """
    Null hypothesis: the model's per-galaxy pairing with the data carries
    no real information. Shuffle which galaxy's parameters (M_bar, R_HI)
    are applied to which galaxy's points, recompute RMS, and see how often
    a random pairing beats the true one.
    """
    rng = np.random.default_rng(seed)
    names = np.array([row["name"] for row in rows])
    unique_names = np.unique(names)
    galaxy_params = {
        n: (rows[np.where(names == n)[0][0]]["MHI"], rows[np.where(names == n)[0][0]]["L36"],
            rows[np.where(names == n)[0][0]]["RHI"])
        for n in unique_names
    }

    true_rms = rms(Vobs, V_pred)
    r = np.array([row["r"] for row in rows])
    Vgas = np.array([row["Vgas"] for row in rows])
    Vdisk = np.array([row["Vdisk"] for row in rows])
    Vbul = np.array([row["Vbul"] for row in rows])
    V_bar2 = v_bar_squared(Vgas, Vdisk, Vbul)

    beat_count = 0
    all_shuffle_rms = np.empty(n_perm)
    shuffled_names_pool = unique_names.copy()
    for i in range(n_perm):
        rng.shuffle(shuffled_names_pool)
        mapping = dict(zip(unique_names, shuffled_names_pool))
        MHI_shuf = np.array([galaxy_params[mapping[n]][0] for n in names])
        L36_shuf = np.array([galaxy_params[mapping[n]][1] for n in names])
        RHI_shuf = np.array([galaxy_params[mapping[n]][2] for n in names])
        M_bar_shuf = baryonic_mass(MHI_shuf, L36_shuf)
        x_shuf = r / RHI_shuf
        V_pred2_shuf = v_pred_squared(V_bar2, M_bar_shuf, x_shuf, clip_negative_vbar2=True)
        shuf_rms = rms(Vobs, np.sqrt(V_pred2_shuf))
        all_shuffle_rms[i] = shuf_rms
        if shuf_rms <= true_rms:
            beat_count += 1

    best_shuffle_rms = float(all_shuffle_rms.min())
    p_value = beat_count / n_perm
    return true_rms, best_shuffle_rms, beat_count, n_perm, p_value, all_shuffle_rms


def main():
    rows, galaxy_table, excluded = load_data()
    Vobs, V_pred, V_bar = compute_predictions(rows)

    this_work_rms = rms(Vobs, V_pred)
    baryons_only_rms = rms(Vobs, V_bar)
    this_work_r2 = r_squared(Vobs, V_pred)

    print("\n=== HEADLINE RESULTS ===")
    print(f"N galaxies retained : {len(galaxy_table)}")
    print(f"N points            : {len(rows)}")
    print(f"This work  RMS      : {this_work_rms:.4f} km/s")
    print(f"Baryons-only RMS    : {baryons_only_rms:.4f} km/s")
    print(f"This work  R^2      : {this_work_r2:.4f}")
    print("\nExpected (paper, results/VERIFIED_RESULTS.txt): "
          "RMS=32.0117 km/s, baryons-only=43.9003 km/s, R^2~0.8666")

    print("\n=== PERMUTATION TEST (this can take a couple of minutes) ===")
    true_rms, best_shuf, beat, n_perm, p, _ = permutation_test(rows, Vobs, V_pred)
    print(f"True RMS: {true_rms:.4f}  Best of {n_perm} shuffles: {best_shuf:.4f}  "
          f"{beat}/{n_perm} shuffles beat the true pairing  p<{max(1/n_perm, p):.1e}")


if __name__ == "__main__":
    main()
