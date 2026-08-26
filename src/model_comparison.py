"""
model_comparison.py — compare this work against three benchmark models on
the same 171-galaxy, N=3346-point SPARC sample:

    1. Baryons only        (0 free global params: just V_bar)
    2. RAR                 (McGaugh, Lelli & Schombert 2016; 0 free params,
                             fixed g_dagger = 1.2e-10 m/s^2)
    3. MOND, simple IF      (fixed a0 = 1.2e-10 m/s^2, 0 free params)
    4. This work            (3 frozen global params: A0, n, x0 — see model.py)
    5. NFW, per-galaxy fit  (2 params/galaxy x 171 galaxies = 342 params,
                             multi-start nonlinear least squares)

Reports RMS, AIC, and BIC for each. AIC/BIC use
    AIC = N*ln(RSS/N) + 2k
    BIC = N*ln(RSS/N) + k*ln(N)
with N = number of points (3346) and k = number of free parameters
*fit to this sample* (0 for the fixed-constant benchmarks, 3 for this
work, 342 for per-galaxy NFW).

IMPORTANT, stated plainly per the paper's own scoping: NFW here is an
UNCONSTRAINED per-galaxy benchmark (each galaxy's halo fit independently,
with no cosmological prior tying its parameters to the others). This is
the standard "best possible fit" comparison, not a cosmologically-
constrained LCDM prediction — see the paper's Discussion for that caveat.
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).parent))
from model import v_pred_squared, v_bar_squared, baryonic_mass, A0, N_EXP, X0  # noqa: E402
from sparc_validation import load_data, rms  # noqa: E402

G_DAGGER = 1.2e-10   # m/s^2, McGaugh, Lelli & Schombert 2016
A0_MOND = 1.2e-10    # m/s^2, standard literature value (fixed, not refit)
KPC_TO_M = 3.0856775814913673e19
KM_TO_M = 1000.0
NFW_SEEDS = [1, 7, 42, 99, 2026, 7, 13, 31, 55, 88, 111, 202, 314, 512, 999, 2027]


def _g_bar(V_bar2_km2_s2, r_kpc):
    """Convert V_bar^2 [km^2/s^2], r [kpc] -> baryonic acceleration [m/s^2]."""
    V_bar2_m2_s2 = np.clip(V_bar2_km2_s2, 0, None) * KM_TO_M ** 2
    r_m = r_kpc * KPC_TO_M
    return V_bar2_m2_s2 / r_m


def rar_prediction(V_bar2, r_kpc, g_dagger=G_DAGGER):
    """McGaugh, Lelli & Schombert (2016) radial acceleration relation."""
    g_bar = _g_bar(V_bar2, r_kpc)
    with np.errstate(over="ignore", invalid="ignore"):
        g_obs = g_bar / (1 - np.exp(-np.sqrt(g_bar / g_dagger)))
    g_obs = np.where(g_bar
