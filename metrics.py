"""
metrics.py - Statistical Metrics & Likelihood Functions
=======================================================
Computes reduced chi-squared, log-residuals, and cross-validation statistics.
"""

import numpy as np

def calculate_residuals(g_obs, g_model):
    """
    Computes logarithmic acceleration residuals:
        Delta log_10(g) = log_10(g_obs) - log_10(g_model)
    """
    g_o = np.asarray(g_obs, dtype=np.float64)
    g_m = np.asarray(g_model, dtype=np.float64)
    mask = (g_o > 0.0) & (g_m > 0.0)
    res = np.zeros_like(g_o)
    res[mask] = np.log10(g_o[mask]) - np.log10(g_m[mask])
    return res[mask]

def calculate_reduced_chi2(v_obs, v_model, v_err, n_params=0):
    """
    Calculates reduced chi-squared:
        chi^2_red = (1 / (N - k)) * sum((v_obs - v_model)^2 / v_err^2)
    """
    v_o = np.asarray(v_obs, dtype=np.float64)
    v_m = np.asarray(v_model, dtype=np.float64)
    err = np.maximum(np.asarray(v_err, dtype=np.float64), 1e-4)
    
    dof = len(v_o) - n_params
    if dof <= 0:
        dof = 1
    chi2 = np.sum(((v_o - v_m) / err)**2)
    return float(chi2 / dof)
