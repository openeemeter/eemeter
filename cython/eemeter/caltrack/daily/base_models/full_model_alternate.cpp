#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

//Define boundary limits for exponential smoothing
double min_pos_system_value = std::sqrt(std::numeric_limits<double>::denorm_min() * 1e20);
double max_pos_system_value = std::sqrt(std::numeric_limits<double>::max() * 1e-20);
double ln_min_pos_system_value = std::log(min_pos_system_value);
double ln_max_pos_system_value = std::log(max_pos_system_value);

//static function for encapsulation and faster runtimes
static PyObject* full_model(
    double hdd_bp,
    double hdd_beta,
    double hdd_k,
    double cdd_bp,
    double cdd_beta,
    double cdd_k,
    double intercept,
    const double* T_fit_bnds,
    const double* T,
    const int T_size) {

    npy_intp dims[1] = {T_size};

    std::vector<double> E_tot;
    E_tot.reserve(T_size);
    // if all variables are zero, return tidd model
    if ((hdd_beta == 0) && (cdd_beta == 0)) {
        std::fill_n(std::back_inserter(E_tot), T_size, intercept);
        return PyArray_SimpleNewFromData(
            1, dims, NPY_DOUBLE, &E_tot[0]);
    }

    auto [T_min_it, T_max_it] = std::minmax_element(T_fit_bnds, T_fit_bnds + 2);
    double T_min = *T_min_it;
    double T_max = *T_max_it;

    if (cdd_bp < hdd_bp) {
        std::swap(hdd_bp, cdd_bp);
        std::swap(hdd_beta, cdd_beta);
        std::swap(hdd_k, cdd_k);
    }
    
    std::transform(T, T + T_size, std::back_inserter(E_tot), [&](double Ti) {
        double T_bp, beta, k;

        if ((Ti < hdd_bp) || ((hdd_bp == cdd_bp) && (cdd_bp >= T_max))) {
            // Temperature is within the heating model
            T_bp = hdd_bp;
            beta = -hdd_beta;
            k = hdd_k;
        } else if ((Ti > cdd_bp) || ((hdd_bp == cdd_bp) && (hdd_bp <= T_min))) {
            // Temperature is within the cooling model
            T_bp = cdd_bp;
            beta = cdd_beta;
            k = -cdd_k;
        } else {
            // Temperature independent
            beta = 0;
        }

        // Evaluate
        if (beta == 0) {  // tidd
            return intercept;
        } else if (k == 0) {  // c_hdd
            return beta * (Ti - T_bp) + intercept;
        } else {  // smoothed c_hdd
            double c_hdd = beta * (Ti - T_bp) + intercept;

            double exp_interior = 1 / k * (Ti - T_bp);
            exp_interior = std::clamp(
                exp_interior,
                ln_min_pos_system_value,
                ln_max_pos_system_value);
            return std::abs(beta * k) * (std::exp(exp_interior) - 1) + c_hdd;
        }
    });

    return PyArray_SimpleNewFromData(
        1, dims, NPY_DOUBLE, &E_tot[0]);
}