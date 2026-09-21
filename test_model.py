"""
test_model.py - Unit Tests: Relativistic Boundary Stress Tensor
==============================================================
Verifies theoretical limits, numerical consistency, and zero-free-parameter locks.
"""

import sys
import os
import numpy as np

from model import g_obs_boundary, g_obs_mond, v_from_g, A_STAR
from metrics import calculate_residuals, calculate_reduced_chi2

def test_asymptotic_newtonian_limit():
    """For g_bar >> a_star, g_obs must approach g_bar (Newtonian UV limit)."""
    g_high = np.array([1e-6, 1e-5, 1e-4])  # >> 1.2e-10
    g_pred = g_obs_boundary(g_high)
    rel_error = np.abs(g_pred - g_high) / g_high
    assert np.all(rel_error < 1e-4), f"Failed Newtonian limit: {rel_error}"

def test_asymptotic_deep_mond_limit():
    """For g_bar << a_star, g_obs must approach sqrt(g_bar * a_star) (Deep IR limit)."""
    g_low = np.array([1e-13, 1e-14, 1e-15])  # << 1.2e-10
    g_pred = g_obs_boundary(g_low)
    g_mond_expected = np.sqrt(g_low * A_STAR)
    rel_error = np.abs(g_pred - g_mond_expected) / g_mond_expected
    assert np.all(rel_error < 0.05), f"Failed Deep IR limit: {rel_error}"

def test_parameter_immutability():
    """Verifies that the canonical acceleration scale is locked."""
    assert A_STAR == 1.204e-10, "Canonical scale parameter drift detected!"

def test_zero_and_negative_inputs():
    """Verifies numerical stability on boundary edge cases."""
    assert g_obs_boundary(0.0) == 0.0
    assert g_obs_boundary(-1.0) == 0.0

def test_velocity_conversion():
    """Tests v = sqrt(r * g) scaling."""
    r_kpc = 10.0
    g = 1e-10
    v = v_from_g(g, r_kpc)
    assert 170.0 < v < 180.0, f"Unexpected velocity: {v}"

def test_residuals_and_chi2():
    """Tests statistical metric functions."""
    g_o = np.array([1e-10, 2e-10, 3e-10])
    g_m = np.array([1e-10, 2e-10, 3e-10])
    res = calculate_residuals(g_o, g_m)
    assert np.all(np.abs(res) < 1e-10)
    
    chi2 = calculate_reduced_chi2(np.array([100.0, 150.0]), np.array([100.0, 150.0]), np.array([5.0, 5.0]))
    assert chi2 == 0.0

if __name__ == "__main__":
    test_asymptotic_newtonian_limit()
    test_asymptotic_deep_mond_limit()
    test_parameter_immutability()
    test_zero_and_negative_inputs()
    test_velocity_conversion()
    test_residuals_and_chi2()
    print("[+] All unit tests PASSED successfully!")
