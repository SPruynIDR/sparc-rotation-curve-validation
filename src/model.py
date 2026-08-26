"""
model.py — Global Baryonic Boundary Scaling Relation for galaxy rotation curves.

This module is the SINGLE SOURCE OF TRUTH for the model's mathematical form
and frozen constants. Every other script in this repository imports from
here rather than re-implementing the formula, so there is exactly one place
that can be wrong.

Model
-----
    V_pred(r)^2 = V_bar(r)^2 + [ A0 * M_bar^(1/4) * F(x) ]^2

    F(x) = x^n / (x^n + x0^n),   x = r / R_HI

    V_bar(r)^2 = sign(V_gas) V_gas^2 + sign(V_disk) V_disk^2 + sign(V_bul) V_bul^2

    M_bar = 1.33 * M_HI + Upsilon * L_[3.6]

Two points to get exactly right, both of which caused real reproduction
failures during development and are recorded here so they are not
re-broken:

1. Upsilon (the stellar mass-to-light ratio) scales ONLY the mass term
   M_bar. It must NEVER be applied to V_disk or V_bul. Applying it to the
   velocity components instead of the mass amplitude silently changes
   every downstream number and does not raise an error.

2. V_bar^2 is built additively in QUADRATURE from the three native
   SPARC-tabulated velocity components, each with its SIGN PRESERVED
   before squaring (SPARC stores V_gas, V_disk, V_bul as signed
   quantities; a small number of points have negative V_gas). The
   boundary term is added to V_bar^2 in quadrature, not to V_bar itself.
"""

from __future__ import annotations
import numpy as np

# ---------------------------------------------------------------------------
# Frozen constants (global fit to N=3346 points across 171 SPARC galaxies,
# see sparc_validation.py for how these were obtained). Units: A0 is in
# km/s * Msun^(-1/4), with M_bar in units of 1e9 Msun (SPARC's native units).
# ---------------------------------------------------------------------------
A0 = 0.339006
N_EXP = 2.01362
X0 = 0.44006
UPSILON = 0.5          # Msun/Lsun, applied to L_[3.6] only
HI_HELIUM_FACTOR = 1.33  # standard correction for primordial He, applied to M_HI


def boundary_profile(x: np.ndarray, n: float = N_EXP, x0: float = X0) -> np.ndarray:
    """F(x) = x^n / (x^n + x0^n). x = r / R_HI, dimensionless."""
    x = np.asarray(x, dtype=float)
    xn = x ** n
    return xn / (xn + x0 ** n)


def baryonic_mass(M_HI: np.ndarray, L36: np.ndarray, upsilon: float = UPSILON) -> np.ndarray:
    """M_bar = 1.33 * M_HI + Upsilon * L_[3.6]. Same units in and out (1e9 Msun)."""
    return HI_HELIUM_FACTOR * np.asarray(M_HI, dtype=float) + upsilon * np.asarray(L36, dtype=float)


def v_bar_squared(V_gas: np.ndarray, V_disk: np.ndarray, V_bul: np.ndarray) -> np.ndarray:
    """Sign-preserved quadrature sum of the three native SPARC velocity components."""
    V_gas = np.asarray(V_gas, dtype=float)
    V_disk = np.asarray(V_disk, dtype=float)
    V_bul = np.asarray(V_bul, dtype=float)
    return (
        np.sign(V_gas) * V_gas ** 2
        + np.sign(V_disk) * V_disk ** 2
        + np.sign(V_bul) * V_bul ** 2
    )


def v_pred_squared(
    V_bar2: np.ndarray,
    M_bar: np.ndarray,
    x: np.ndarray,
    A0: float = A0,
    n: float = N_EXP,
    x0: float = X0,
    clip_negative_vbar2: bool = True,
) -> np.ndarray:
    """
    Full model prediction V_pred^2 = V_bar^2 + [A0 * M_bar^0.25 * F(x)]^2.

    clip_negative_vbar2: the headline-convention behaviour (N=3346 points).
    A small number of points (2, both in UGC01281) have V_bar^2 < 0 from a
    net-negative gas contribution; the original methodology clips these to
    zero rather than dropping them. Set False to instead drop them
    (N=3344), which is also a legitimate, previously-used convention — see
    results/VERIFIED_RESULTS.txt for both numbers.
    """
    V_bar2 = np.asarray(V_bar2, dtype=float).copy()
    if clip_negative_vbar2:
        V_bar2[V_bar2 < 0] = 0.0
    F = boundary_profile(x, n=n, x0=x0)
    boundary_term = A0 * np.asarray(M_bar, dtype=float) ** 0.25 * F
    return V_bar2 + boundary_term ** 2


def v_pred(V_gas, V_disk, V_bul, M_HI, L36, r, R_HI, **kwargs) -> np.ndarray:
    """Convenience wrapper: raw SPARC columns in, predicted V(r) out (km/s)."""
    V_bar2 = v_bar_squared(V_gas, V_disk, V_bul)
    M_bar = baryonic_mass(M_HI, L36)
    x = np.asarray(r, dtype=float) / np.asarray(R_HI, dtype=float)
    return np.sqrt(v_pred_squared(V_bar2, M_bar, x, **kwargs))
