import pandas as pd
"""
Statistical Evaluation & Model Comparison Metrics
=================================================
Calculates chi-squared, reduced chi-squared, log residuals, and comparative benchmarks.
"""

import numpy as np

def compute_galaxy_metrics(v_obs, v_err, v_model, g_obs, g_model, n_params=0):
    """
    Compute statistical goodness-of-fit metrics for a galaxy rotation curve.
    """
    v_obs = np.asarray(v_obs, dtype=np.float64)
    v_err = np.maximum(np.asarray(v_err, dtype=np.float64), 0.1)
    v_model = np.asarray(v_model, dtype=np.float64)
    
    n_pts = len(v_obs)
    dof = max(n_pts - n_params, 1)
    
    residuals_v = (v_obs - v_model) / v_err
    chi2 = np.sum(residuals_v ** 2)
    reduced_chi2 = chi2 / dof
    
    # Log acceleration residual
    log_res = np.log10(np.maximum(g_obs, 1e-20)) - np.log10(np.maximum(g_model, 1e-20))
    rms_log_res = np.sqrt(np.mean(log_res ** 2))
    
    return {
        "n_pts": n_pts,
        "dof": dof,
        "chi2": chi2,
        "reduced_chi2": reduced_chi2,
        "mean_log_res": np.mean(log_res),
        "rms_log_res": rms_log_res,
        "log_residuals": log_res
    }

def compute_sample_summary(df, a_star=1.204e-10):
    """
    Compute full sample statistics across all galaxies in dataframe.
    """
    from sparc_benchmark.model import predict_rotation_curve, mond_acceleration, velocity_from_g, KPC_TO_METERS, KM_S_TO_M_S
    
    results = []
    all_log_res_bst = []
    all_log_res_mond = []
    all_chi2_bst = []
    all_chi2_mond = []
    
    for gid, group in df.groupby("galaxy"):
        r = group["r_kpc"].values
        v_o = group["v_obs"].values
        v_e = group["v_err"].values
        v_g = group["v_gas"].values
        v_d = group["v_disk"].values
        v_b = group["v_bulge"].values
        
        # Boundary stress tensor model (0 free params)
        v_m_bst, g_bar, g_m_bst = predict_rotation_curve(r, v_g, v_d, v_b, a_star=a_star)
        
        # Observed g_obs from v_obs
        r_m = r * KPC_TO_METERS
        g_obs_data = ((v_o * KM_S_TO_M_S) ** 2) / r_m
        
        met_bst = compute_galaxy_metrics(v_o, v_e, v_m_bst, g_obs_data, g_m_bst, n_params=0)
        
        # MOND model (0 free params per galaxy, fixed a0)
        g_m_mond = mond_acceleration(g_bar, function_type='simple')
        v_m_mond = velocity_from_g(r, g_m_mond)
        met_mond = compute_galaxy_metrics(v_o, v_e, v_m_mond, g_obs_data, g_m_mond, n_params=0)
        
        all_log_res_bst.extend(met_bst["log_residuals"])
        all_log_res_mond.extend(met_mond["log_residuals"])
        all_chi2_bst.append(met_bst["reduced_chi2"])
        all_chi2_mond.append(met_mond["reduced_chi2"])
        
        results.append({
            "galaxy": gid,
            "morphology": group["morphology"].iloc[0],
            "n_pts": met_bst["n_pts"],
            "red_chi2_bst": met_bst["reduced_chi2"],
            "red_chi2_mond": met_mond["reduced_chi2"],
            "rms_bst": met_bst["rms_log_res"],
            "rms_mond": met_mond["rms_log_res"]
        })
        
    res_df = pd.DataFrame(results)
    
    summary = {
        "n_galaxies": len(results),
        "total_data_points": len(all_log_res_bst),
        "bst_median_reduced_chi2": float(np.median(all_chi2_bst)),
        "bst_mean_reduced_chi2": float(np.mean(all_chi2_bst)),
        "bst_rms_scatter_dex": float(np.sqrt(np.mean(np.array(all_log_res_bst)**2))),
        "mond_median_reduced_chi2": float(np.median(all_chi2_mond)),
        "mond_mean_reduced_chi2": float(np.mean(all_chi2_mond)),
        "mond_rms_scatter_dex": float(np.sqrt(np.mean(np.array(all_log_res_mond)**2))),
        "res_df": res_df
    }
    return summary
