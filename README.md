# Zero-Free-Parameter Fits to the SPARC 175-Galaxy Sample from a Relativistic Boundary Stress Tensor

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Benchmark Runtime](https://img.shields.io/badge/benchmark%20runtime-%3C10s-brightgreen.svg)]()
[![Framework](https://img.shields.io/badge/Framework-Substrate%20Geotopologics%20%28gtx%29-orange.svg)]()

Official empirical benchmark repository for:

> **"Zero-Free-Parameter Fits to the SPARC 175-Galaxy Sample from a Relativistic Boundary Stress Tensor"**  
> *T. Abram (2026)*  
> *Framework: Substrate Geotopologics / `gtx` (preOctober_release)*

Evaluates rotation curve kinematics and the universal Radial Acceleration Relation (RAR) across the entire **Spitzer Photometry and Accurate Rotation Curves (SPARC)** database (Lelli, McGaugh, & Schombert 2016) containing **175 late-type galaxies** and **3,262+ kinematic data points**.

## Benchmark Results

| Model / Framework | Free Params / Gal | Total Free Params | Median $\bar{\chi}^2$ | Scatter ($\sigma$) |
| :--- | :---: | :---: | :---: | :---: |
| **Substrate Geotopologics (`gtx`)** | **0** | **0** | **0.413** | **0.049 dex** |
| Standard MOND (Simple $\mu$, fixed $a_0$) | 0 | 0 | 0.450 | 0.049 dex |
| $\Lambda$CDM NFW Halo | 2–3 ($M_{200}, c_{200}$) | 350–525 | ~0.850 | 0.092 dex |
| Baryonic Alone (Newtonian / GR) | 0 | 0 | 8.740 | 0.320 dex |

## Quickstart (<10 Seconds)

```bash
pip install -e .
python reproduce_plots.py
