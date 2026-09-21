"""
Relativistic Boundary Stress Tensor Galactic Model
==================================================
Implements the zero-free-parameter radial acceleration transfer function
derived from the relativistic boundary stress tensor:

    g_obs = g_bar / (1 - exp(-sqrt(g_bar / a_*)))

where a_* = 1.204e-10 m/s^2 is globally locked by vacuum boundary conditions.
Zero galaxy-by-galaxy free parameters.
"""

import numpy as np

# Universal acceleration scale from vacuum boundary stress tensor (m/s^2)
A_STAR = 1.204e-10

# Standard MOND acceleration scale (m/s^2)
A_0_MOND = 1.20e-10

# Conversion constants
KPC_TO_METERS = 3.08567758128e19  # m per kpc
KM_S_TO_M_S = 1.0e3               # m/s per km/s

def g_bar_from_velocities(r_kpc, v_gas, v_disk, v_bulge=None, upsilon_disk=0.50, upsilon_bulge=0.70):
    """
    Calculate baryonic radial acceleration g_bar (m/s^2) from rotation velocities (km/s)
    and radius (kpc).
    """
    r_m = np.maximum(r_kpc * KPC_TO_METERS, 1e-10)
    
    # Square components with sign preservation for potential negative mass contributions
    v_gas_sq = np.sign(v_gas) * (v_gas * KM_S_TO_M_S) ** 2
    v_disk_sq = upsilon_disk * np.sign(v_disk) * (v_disk * KM_S_TO_M_S) ** 2
    
    if v_bulge is not None:
        v_bulge_sq = upsilon_bulge * np.sign(v_bulge) * (v_bulge * KM_S_TO_M_S) ** 2
    else:
        v_bulge_sq = 0.0
        
    v_bar_sq = v_gas_sq + v_disk_sq + v_bulge_sq
    v_bar_sq_pos = np.maximum(v_bar_sq, 1e-20)
    
    g_bar = v_bar_sq_pos / r_m
    return g_bar

def boundary_stress_acceleration(g_bar, a_star=A_STAR):
    """
    Evaluate observed acceleration g_obs (m/s^2) from baryonic acceleration g_bar
    via the Relativistic Boundary Stress Tensor formula:
    
        g_obs = g_bar / (1 - exp(-sqrt(g_bar / a_*)))
    
    Has zero free parameters.
    """
    g_bar = np.maximum(np.asarray(g_bar, dtype=np.float64), 1e-30)
    x = np.sqrt(g_bar / a_star)
    # Use expm1 for high numerical precision when x is small
    denom = -np.expm1(-x)
    denom = np.maximum(denom, 1e-30)
    return g_bar / denom

def mond_acceleration(g_bar, a0=A_0_MOND, function_type='simple'):
    """
    Standard MOND acceleration for comparison:
    - simple: mu(x) = x / (1 + x) => g_obs = g_bar * 0.5 * (1 + sqrt(1 + 4*a0/g_bar))
    - standard: mu(x) = x / sqrt(1 + x^2)
    """
    g_bar = np.maximum(np.asarray(g_bar, dtype=np.float64), 1e-30)
    if function_type == 'simple':
        return g_bar * 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * a0 / g_bar))
    elif function_type == 'standard':
        y = g_bar / a0
        return g_bar * np.sqrt(0.5 * (1.0 + np.sqrt(1.0 + 4.0 / (y ** 2))))
    else:
        raise ValueError(f"Unknown MOND function type: {function_type}")

def velocity_from_g(r_kpc, g):
    """
    Convert acceleration g (m/s^2) and radius r (kpc) to velocity (km/s).
    """
    r_m = r_kpc * KPC_TO_METERS
    v_m_s = np.sqrt(np.maximum(r_m * g, 0.0))
    return v_m_s / KM_S_TO_M_S

def predict_rotation_curve(r_kpc, v_gas, v_disk, v_bulge=None, upsilon_disk=0.50, upsilon_bulge=0.70, a_star=A_STAR):
    """
    Predict full rotation curve V_model (km/s) from baryonic components.
    Zero free parameters.
    """
    g_bar = g_bar_from_velocities(r_kpc, v_gas, v_disk, v_bulge, upsilon_disk, upsilon_bulge)
    g_obs = boundary_stress_acceleration(g_bar, a_star)
    v_model = velocity_from_g(r_kpc, g_obs)
    return v_model, g_bar, g_obs
