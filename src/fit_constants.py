"""
fit_constants.py — independently re-derive A0, n, x0 from raw SPARC data
via global differential-evolution optimization, rather than trusting the
frozen values hardcoded in model.py. Confirms the reproducibility claim in
the paper: 5 independent random seeds converge to the same constants to
better than 1e-6.

This is slow (minutes). It is NOT run by default in sparc_validation.py or
model_comparison.py, which use the frozen, already-validated constants —
run this separately when you want to check the constants themselves,
not just the downstream numbers.

Run:
    python src/fit_constants.py
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

sys.path.insert(0, str(Path(__file__).parent))
from model import v_bar_squared, baryonic_mass, v_pred_squared  # noqa: E402
from sparc_validation import load_data, rms  # noqa: E402

SEEDS = [1, 7, 42, 99, 2026]
BOUNDS = [(0.01, 2.0), (0.2, 8.0), (0.01, 3.0)]  # A0, n, x0


def build_objective(Vobs, V_bar2, M_bar, x):
    def objective(params):
        A0, n, x0 = params
        V_pred2 = v_pred_squared(V_bar2, M_bar, x, A0=A0, n=n, x0=x0, clip_negative_vbar2=True)
        return rms(Vobs, np.sqrt(np.clip(V_pred2, 0, None)))
    return objective


def main():
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
    objective = build_objective(Vobs, V_bar2, M_bar, x)

    print(f"{'seed':>6}{'A0':>14}{'n':>12}{'x0':>12}{'RMS':>12}")
    fits = []
    for seed in SEEDS:
        result = differential_evolution(
            objective,
            BOUNDS,
            seed=seed,
            popsize=25,
            maxiter=3000,
            strategy="best1bin",
            mutation=(0.5, 1),
            recombination=0.7,
            tol=1e-12,
            polish=True,
        )
        A0_fit, n_fit, x0_fit = result.x
        fits.append(result.x)
        print(f"{seed:>6}{A0_fit:>14.6f}{n_fit:>12.5f}{x0_fit:>12.5f}{result.fun:>12.6f}")

    fits = np.array(fits)
    spread = fits.max(axis=0) - fits.min(axis=0)
    print(f"\nSpread across {len(SEEDS)} seeds (A0, n, x0): {spread}")
    print("Expected frozen values: A0=0.339006, n=2.01362, x0=0.44006 (see src/model.py)")


if __name__ == "__main__":
    main()
