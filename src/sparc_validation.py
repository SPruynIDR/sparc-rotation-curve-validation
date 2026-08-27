"""
sparc_validation.py — reproduce the headline numbers of the Paper II
rotation-curve validation from raw, public SPARC data.

This version reads both tables by fixed column POSITION (whitespace-split),
using the exact column order confirmed from a real successful download
(see the comments below) rather than relying on astropy's CDS auto-detect
or a byte-offset regex parser — both of which proved fragile in practice.
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent.parent / "data"
PERMUTATION_SEED = 20260710
N_PERMUTATIONS = 10_000

sys.path.insert(0, str(Path(__file__).parent))
from model import v_pred_squared, v_bar_squared, baryonic_mass  # noqa: E402

# Confirmed real column order (from an actual successful download's printed
# header list):
# Table1: Galaxy(0) T(1) D(2) e_D(3) f_D(4) Inc(5) e_Inc(6) L[3.6](7)
#         e_L[3.6](8) Reff(9) SBeff(10) Rdisk(11) SBdisk(12) MHI(13)
#         RHI(14) Vflat(15) e_Vflat(16) Q(17) Ref.(18)
# Table2: ID(0) D(1) R(2) Vobs(3) e_Vobs(4) Vgas(5) Vdisk(6) Vbul(7)
#         SBdisk(8) SBbul(9)
T1_IDX = dict(name=0, T=1, L36=7, MHI=13, RHI=14, Q=17)
T2_IDX = dict(name=0, r=2, Vobs=3, Vgas=5, Vdisk=6, Vbul=7)


def _read_fixed_position(path: Path, idx_map: dict) -> dict:
    """Skip the CDS header (everything up to and including the LAST line
    of dashes), then split each data line on whitespace and pull columns
    by confirmed position."""
    lines = path.read_text(errors="replace").splitlines()
    dash_lines = [i for i, ln in enumerate(lines) if set(ln.strip()) == {"-"} and len(ln.strip()) > 10]
    if not dash_lines:
        raise ValueError(f"{path.name}: could not find the CDS header separator line")
    data_start = dash_lines[-1] + 1

    cols = {key: [] for key in idx_map}
    max_idx = max(idx_map.values())
    for ln in lines[data_start:]:
        if not ln.strip():
            continue
        parts = ln.split()
        if len(parts) <= max_idx:
            continue  # malformed/short line, skip rather than crash
        for key, i in idx_map.items():
            cols[key].append(parts[i])
    return cols


def _to_float(values):
    out = []
    for v in values:
        try:
            out.append(float(v))
        except ValueError:
            out.append(np.nan)
    return np.array(out)


def load_data(data_dir: Path = DATA_DIR):
    t1_path = data_dir / "SPARC_Lelli2016c.mrt"
    t2_path = data_dir / "MassModels_Lelli2016c.mrt"
    if not t1_path.exists() or not t2_path.exists():
        raise FileNotFoundError(
            f"Expected {t1_path.name} and {t2_path.name} in {data_dir}. "
            f"Run `python src/download_data.py` first."
        )

    t1 = _read_fixed_position(t1_path, T1_IDX)
    t2 = _read_fixed_position(t2_path, T2_IDX)

    names1 = np.array([s.strip() for s in t1["name"]])
    RHI = _to_float(t1["RHI"])
    L36 = _to_float(t1["L36"])
    MHI = _to_float(t1["MHI"])
    T = _to_float(t1["T"])

    print(f"[info] Table1 parsed: {len(names1)} galaxies. "
          f"Sample: {list(zip(names1[:3], RHI[:3], L36[:3], MHI[:3]))}")

    good = RHI > 0
    excluded_names = names1[~good]
    print(f"[info] excluding {(~good).sum()} galaxies with R_HI<=0: {list(excluded_names)}")

    galaxy_table = {
        name: {"RHI": rhi, "L36": l36, "MHI": mhi, "T": t}
        for name, rhi, l36, mhi, t, keep in zip(names1, RHI, L36, MHI, T, good)
        if keep
    }

    names2 = np.array([s.strip() for s in t2["name"]])
    r = _to_float(t2["r"])
    Vobs = _to_float(t2["Vobs"])
    Vgas = _to_float(t2["Vgas"])
    Vdisk = _to_float(t2["Vdisk"])
    Vbul = _to_float(t2["Vbul"])

    print(f"[info] Table2 parsed: {len(names2)} points. "
          f"Unique galaxy names in Table2: {len(np.unique(names2))}")

    keep_point = np.array([n in galaxy_table for n in names2])
    print(f"[info] {keep_point.sum()} of {len(names2)} points retained "
          f"(target: 3346 across 171 galaxies)")

    rows = []
    for i in np.where(keep_point)[0]:
        g = galaxy_table[names2[i]]
        rows.append(dict(
            name=names2[i], r=r[i], Vobs=Vobs[i], Vgas=Vgas[i],
            Vdisk=Vdisk[i], Vbul=Vbul[i],
            RHI=g["RHI"], L36=g["L36"], MHI=g["MHI"], T=g["T"],
        ))
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
    V_bar = np.sqrt(np.clip(V_bar2, 0, None))

    return Vobs, V_pred, V_bar


def rms(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def r_squared(obs, pred):
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    return float(1 - ss_res / ss_tot)


def permutation_test(rows, Vobs, V_pred, seed=PERMUTATION_SEED, n_perm=N_PERMUTATIONS):
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
    pool = unique_names.copy()
    for i in range(n_perm):
        rng.shuffle(pool)
        mapping = dict(zip(unique_names, pool))
        MHI_s = np.array([galaxy_params[mapping[n]][0] for n in names])
        L36_s = np.array([galaxy_params[mapping[n]][1] for n in names])
        RHI_s = np.array([galaxy_params[mapping[n]][2] for n in names])
        M_bar_s = baryonic_mass(MHI_s, L36_s)
        x_s = r / RHI_s
        V_pred2_s = v_pred_squared(V_bar2, M_bar_s, x_s, clip_negative_vbar2=True)
        shuf_rms = rms(Vobs, np.sqrt(V_pred2_s))
        all_shuffle_rms[i] = shuf_rms
        if shuf_rms <= true_rms:
            beat_count += 1

    return true_rms, float(all_shuffle_rms.min()), beat_count, n_perm, beat_count / n_perm, all_shuffle_rms


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
    print("\nExpected: RMS=32.0117 km/s, baryons-only=43.9003 km/s, R^2~0.8666")

    print("\n=== PERMUTATION TEST ===")
    true_rms, best_shuf, beat, n_perm, p, _ = permutation_test(rows, Vobs, V_pred)
    print(f"True RMS: {true_rms:.4f}  Best of {n_perm} shuffles: {best_shuf:.4f}  "
          f"{beat}/{n_perm} beat the true pairing  p<{max(1/n_perm, p):.1e}")


if __name__ == "__main__":
    main()
