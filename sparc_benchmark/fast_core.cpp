/**
 * Fast C++ Core Kernel for Relativistic Boundary Stress Tensor Galactic Acceleration
 * ==================================================================================
 * High-throughput vectorized computation for SPARC kinematic catalogs.
 * Compile: g++ -O3 -shared -fPIC -std=c++17 fast_core.cpp -o libfast_boundary.so
 */

#include <cmath>
#include <vector>
#include <iostream>
#include <chrono>

constexpr double A_STAR = 1.204e-10;             // m/s^2 (locked vacuum boundary scale)
constexpr double KPC_TO_METERS = 3.08567758128e19;
constexpr double KM_S_TO_M_S = 1.0e3;

extern "C" {

/**
 * Direct point-wise calculation of observed acceleration from baryonic acceleration
 */
double boundary_stress_acceleration_scalar(double g_bar, double a_star = A_STAR) {
    if (g_bar <= 1e-30) return 0.0;
    double x = std::sqrt(g_bar / a_star);
    double denom = -std::expm1(-x);
    return g_bar / std::max(denom, 1e-30);
}

/**
 * Batch processing of rotation curve profiles
 */
void compute_rotation_curve_batch(
    const double* r_kpc,
    const double* v_gas,
    const double* v_disk,
    const double* v_bulge,
    double upsilon_disk,
    double upsilon_bulge,
    double* v_model_out,
    double* g_obs_out,
    int n_points,
    double a_star = A_STAR
) {
    for (int i = 0; i < n_points; ++i) {
        double r_m = std::max(r_kpc[i] * KPC_TO_METERS, 1e-10);
        
        double vg = v_gas[i] * KM_S_TO_M_S;
        double vd = v_disk[i] * KM_S_TO_M_S;
        double vb = (v_bulge != nullptr) ? (v_bulge[i] * KM_S_TO_M_S) : 0.0;
        
        double v_bar_sq = std::copysign(vg * vg, vg) + 
                          upsilon_disk * std::copysign(vd * vd, vd) + 
                          upsilon_bulge * std::copysign(vb * vb, vb);
        v_bar_sq = std::max(v_bar_sq, 1e-20);
        
        double g_bar = v_bar_sq / r_m;
        double x = std::sqrt(g_bar / a_star);
        double denom = std::max(-std::expm1(-x), 1e-30);
        double g_obs = g_bar / denom;
        
        g_obs_out[i] = g_obs;
        v_model_out[i] = std::sqrt(std::max(r_m * g_obs, 0.0)) / KM_S_TO_M_S;
    }
}

} // extern "C"

#ifdef STANDALONE_BENCHMARK
int main() {
    std::cout << "[C++ Benchmark Core] Initializing high-speed vector verification...\n";
    constexpr int N = 1000000;
    std::vector<double> g_bar(N, 1.0e-11);
    std::vector<double> g_obs(N, 0.0);
    
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        g_obs[i] = boundary_stress_acceleration_scalar(g_bar[i]);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    
    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "[C++ Benchmark Core] Evaluated " << N << " kinematic points in " 
              << elapsed_ms << " ms (" << (N / (elapsed_ms * 1e3)) << " million pts/sec).\n";
    std::cout << "g_bar = 1.0e-11 m/s^2 => g_obs = " << g_obs[0] << " m/s^2\n";
    return 0;
}
#endif
