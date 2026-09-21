"""
model.py - Boundary Stress Tensor & RAR Transfer Functions
==========================================================
Implements the macroscopic radial acceleration transfer function:

    g_obs = g_bar / [1 - exp(-sqrt(g_bar / a_*))]

globally locked to a_* = 1.204e-10 m/s^2 with strictly zero free parameters.
"""

import numpy as np

# Globally locked vacuum acoustic boundary acceleration scale
A_STAR = 1.204e-10  # m / s^2
A_MOND = 1.200e-10  # m / s^2 (Milgrom standard)

def g_obs_boundary(g_bar, a_star=A_STAR):
    """
    Computes the macroscopic observed centripetal acceleration from
    the Newtonian baryonic acceleration via the boundary stress tensor.

    Parameters:
        g_bar: float or array_like
            Newtonian baryonic acceleration in m/s^2.
        a_star: float
            Characteristic boundary acceleration scale (default: 1.204e-10 m/s^2).

    Returns:
        g_obs: float or ndarray
            Observed centripetal acceleration.
    """
    g_b = np.asarray(g_bar, dtype=np.float64)
    mask_zero = g_b <= 0.0
    safe_gb = np.where(mask_zero, 1e-30, g_b)

    ratio = np.sqrt(safe_gb / a_star)

    with np.errstate(over='ignore', under='ignore'):
        denominator = 1.0 - np.exp(-ratio)
        denominator = np.where(denominator < 1e-15, ratio, denominator)
        g_out = safe_gb / denominator

    g_out = np.where(mask_zero, 0.0, g_out)
    return g_out if np.ndim(g_bar) > 0 else float(g_out)

def g_obs_mond(g_bar, a0=A_MOND, function="simple"):
    """
    Standard MOND interpolating functions for benchmark comparison.
    'simple' formula: g_obs = g_bar / 2 + sqrt(g_bar^2 / 4 + g_bar * a0)
    """
    g_b = np.asarray(g_bar, dtype=np.float64)
    if function == "simple":
        return 0.5 * g_b + np.sqrt(0.25 * g_b**2 + g_b * a0)
    elif function == "standard":
        ratio = np.sqrt(np.maximum(g_b, 1e-30) / a0)
        denom = 1.0 - np.exp(-ratio)
        return g_b / np.maximum(denom, 1e-15)
    else:
        raise ValueError(f"Unknown MOND function: {function}")

def v_from_g(g, r_kpc):
    """
    Converts centripetal acceleration g (m/s^2) and radius r (kpc)
    to circular velocity V (km/s).
    1 kpc = 3.085677581e19 m
    """
    KPC_TO_M = 3.085677581e19
    r_m = np.asarray(r_kpc, dtype=np.float64) * KPC_TO_M
    v_ms = np.sqrt(np.maximum(g, 0.0) * r_m)
    return v_ms / 1000.0  # km / s
