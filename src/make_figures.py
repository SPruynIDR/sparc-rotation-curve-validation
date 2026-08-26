"""
make_figures.py — regenerate the three figures used in the paper directly
from the raw data and this repository's own model code, so the figures
can never silently drift out of sync with the numbers in the text.

    fig1_parity_residuals.pdf  — V_pred vs V_obs parity plot + residuals
    fig2_boundary_profile.pdf  — F(x) boundary profile collapse
    fig3_permutation.pdf       — permutation-test null distribution

Matplotlib defaults to Type 3 fonts in PDF output, which arXiv's PDF
processing flags; this script forces Type 42 (TrueType) embedding, which
is required for every figure in this project.

Run:
    python src/make_figures.py
"""

from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from model import v_bar_squared, baryonic_mass, v_pred_squared, boundary_profile  # noqa: E402
from sparc_validation import load_data, permutation_test  # noqa: E402

OUT_DIR = Path(__file__).parent.parent / "results" / "figures"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, galaxy_table, excluded = load_data()
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

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(5.5, 6.5), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )
    ax1.scatter(Vobs, V_pred, s=4, alpha=0.35, color="#2b6cb0")
    lims = [0, max(Vobs.max(), V_pred.max()) * 1.05]
    ax1.plot(lims, lims, "k--", lw=1)
    ax1.set_ylabel(r"$V_{\rm pred}$ [km/s]")
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    resid = V_pred - Vobs
    ax2.scatter(Vobs, resid, s=4, alpha=0.35, color="#2b6cb0")
    ax2.axhline(0, color="k", lw=1, ls="--")
    ax2.set_xlabel(r"$V_{\rm obs}$ [km/s]")
    ax2.set_ylabel("residual [km/s]")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig1_parity_residuals.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    xs = np.linspace(0.01, 5, 400)
    ax.plot(xs, boundary_profile(xs), color="#2b6cb0", lw=2, label=r"$F(x)$")
    ax.set_xlabel(r"$x = r / R_{\rm HI}$")
    ax.set_ylabel(r"$F(x)$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_boundary_profile.pdf")
    plt.close(fig)

    print("Running permutation test for Figure 3 (this reuses the same "
          "10,000-shuffle test as sparc_validation.py) ...")
    true_rms, best_shuf, beat, n_perm, p, all_shuffle_rms = permutation_test(rows, Vobs, V_pred)
