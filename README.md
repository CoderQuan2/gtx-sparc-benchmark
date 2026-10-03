SPARC Benchmark Suite: Relativistic Boundary Stress Tensor (gtx)
    
An open-source, reproducible computational benchmark suite for evaluating the Relativistic Boundary Stress Tensor formulation of Substrate Geotopologics (gtx) across the canonical 175-galaxy Spitzer Photometry and Accurate Rotation Curves (SPARC) database (3,262 kinematic points).
Accompanying research manuscript:
Zero-Free-Parameter 
T. Abram (Flint, Michigan, USA — September 2026)
________________


🌌 Overview
Modern observational galactic dynamics presents a persistent dilemma:
* $\Lambda$CDM requires 2–3 free halo parameters per galaxy ($M_{\text{vir}}$, $c$, $\gamma$, totaling 350+ free parameters across SPARC) to match rotation curves, while struggling with the cusp-core discrepancy and diversity problem in dwarf irregulars.
* MOND captures the universal Radial Acceleration Relation (RAR) scaling empirically, but relies on ad-hoc interpolation functions $\mu(x)$ and lacks a universally accepted covariant field theory.
This repository provides an end-to-end implementation of the macroscopic transfer function derived from covariant energy-momentum conservation across the galactic disk boundary hypersurface:
$$g_{\text{obs}} = \frac{g_{\text{bar}}}{1 - \exp\left(-\sqrt{\frac{g_{\text{bar}}}{a_*}}\right)}$$
where the characteristic acceleration scale: $$a_* = 1.204 \times 10^{-10}\ \text{m s}^{-2}$$ is globally locked by vacuum acoustic boundary constraints with strictly zero free parameters per galaxy.
________________


📊 Benchmark Summary
Evaluated across all 175 galaxies (3,262 resolved kinematic data points) in the SPARC catalog:
Model
	Parameters per Galaxy
	Total Free Parameters
	Median Reduced $\bar{\chi}^2$
	Residual Scatter ($\sigma$)
	Execution Time
	Boundary Stress Tensor (gtx)
	0
	0
	0.413
	0.049 dex
	< 3 seconds
	Standard MOND (Simple $\mu$)
	0
	0
	0.450
	0.049 dex
	< 3 seconds
	$\Lambda$CDM (NFW Halos)
	2–3 ($M_{\text{vir}}, c$)
	>350
	0.850
	>0.110 dex
	Minutes / Hours
	________________


⚡ Quickstart & Installation
git clone https://github.com/CoderQuan2/sparc-benchmark.git
cd sparc-benchmark
pip install -e .
python reproduce_plots.py
(Executes in ~2.7 seconds on an 8-core CPU).
________________


📜 Citation
@article{abram2026sparc,
  title={Zero-Free-Parameter Fits to the SPARC 175-Galaxy Sample from a Relativistic Boundary Stress Tensor},
  author={Abram, T.},
  journal={Substrate Geotopologics Technical Preprint Series},
  year={2026},
  month={September},
  note={Flint, Michigan, USA. Contact: tabram@gth-physics.org}
}
________________


⚖️ License
Released under the MIT License.
