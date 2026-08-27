"""
make_figures.py — regenerate the three figures used in the paper directly
from the raw data and this repository's own model code.

This version prints a clear status line after every step and fails loudly
with a full traceback if anything goes wrong, rather than exiting silently.
"""

from __future__ import annotations
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

OUT_DIR = Path(__file__).parent.parent / "results" / "figures"


def main():
    print("[step 1] Importing model + sparc_validation modules...", flush=True)
    from model import v_bar_squared, baryonic_mass, v_pred_squared, boundary_profile
    from sparc_validation import load_data, permutation_test
    print("[step 1] OK", flush=True)

    print(f"[step 2] Creating output directory: {OUT_DIR}", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[step 2] OK, exists={OUT_DIR.exists()}", flush=True)

    print("[step 3] Loading real SPARC data...", flush=True)
    rows, galaxy_table, excluded = load_data()
    print(f"[step 3] OK: {len(galaxy_table)} galaxies, {len(rows)} points", flush=True)

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
    V_pred = np.sqrt(np.clip(v_pred_squared(V_bar2, M_bar, x), 0, None))
    print(f"[step 4] Predictions computed. RMS={np.sqrt(np.mean((Vobs-V_pred)**2)):.4f}", flush=True)

    print("[step 5] Building Figure 1 (parity plot)...", flush=True)
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(5.5, 6.5), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )
    ax1.scatter(Vobs, V_pred, s=4, alpha=0.35, color="#2b6cb0")
    lims = [0, max(Vobs.max(), V_pred.max()) * 1.05]
    ax1.plot(lims, lims, "k--", lw=1)
    ax1.set_ylabel(r"$V_{\rm pred}$ [km/s]")
    ax1.set_xlim(lims); ax1.set_ylim(lims)
    resid = V_pred - Vobs
    ax2.scatter(Vobs, resid, s=4, alpha=0.35, color="#2b6cb0")
    ax2.axhline(0, color="k", lw=1, ls="--")
    ax2.set_xlabel(r"$V_{\rm obs}$ [km/s]")
    ax2.set_ylabel("residual [km/s]")
    fig.tight_layout()
    out1 = OUT_DIR / "fig1_parity_residuals.pdf"
    fig.savefig(out1)
    plt.close(fig)
    print(f"[step 5] OK: wrote {out1}, exists={out1.exists()}, size={out1.stat().st_size if out1.exists() else 'N/A'}", flush=True)

    print("[step 6] Building Figure 2 (boundary profile)...", flush=True)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    xs = np.linspace(0.01, 5, 400)
    ax.plot(xs, boundary_profile(xs), color="#2b6cb0", lw=2, label=r"$F(x)$")
    ax.set_xlabel(r"$x = r / R_{\rm HI}$")
    ax.set_ylabel(r"$F(x)$")
    ax.legend()
    fig.tight_layout()
    out2 = OUT_DIR / "fig2_boundary_profile.pdf"
    fig.savefig(out2)
    plt.close(fig)
    print(f"[step 6] OK: wrote {out2}, exists={out2.exists()}", flush=True)

    print("[step 7] Running permutation test for Figure 3 (this takes ~20s)...", flush=True)
    true_rms, best_shuf, beat, n_perm, p, all_shuffle_rms = permutation_test(rows, Vobs, V_pred)
    print(f"[step 7] Permutation test done: true={true_rms:.4f}, best_shuffle={best_shuf:.4f}", flush=True)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.hist(all_shuffle_rms, bins=60, color="#2b6cb0", alpha=0.75, label=f"{n_perm} shuffled pairings")
    ax.axvline(true_rms, color="crimson", lw=2, label=f"true pairing ({true_rms:.2f} km/s)")
    ax.set_xlabel("RMS [km/s]")
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    out3 = OUT_DIR / "fig3_permutation.pdf"
    fig.savefig(out3)
    plt.close(fig)
    print(f"[step 7] OK: wrote {out3}, exists={out3.exists()}", flush=True)

    print(f"\n[DONE] Contents of {OUT_DIR}:", flush=True)
    for f in sorted(OUT_DIR.iterdir()):
        print(f"  {f.name}  ({f.stat().st_size} bytes)", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n[FATAL ERROR] make_figures.py crashed. Full traceback:", flush=True)
        traceback.print_exc()
        sys.exit(1)
