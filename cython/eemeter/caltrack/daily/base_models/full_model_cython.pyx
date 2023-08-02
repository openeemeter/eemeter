# distutils: language = c++
# cython: boundscheck=False, wraparound=False, nonecheck=False, initializedcheck=False, cdivision=True

cimport numpy as np

cdef extern from "full_model_alternate.cpp":
    cdef np.ndarray[np.double_t, ndim=1] full_model( double hdd_bp, double hdd_beta, double hdd_k,
                                    double cdd_bp, double cdd_beta, double cdd_k,
                                    double intercept, const double* T_fit_bnds, 
                                    const double* T, const int T_size) nogil

def full_model_wrapper(double hdd_bp,double hdd_beta,double hdd_k,double cdd_bp,double cdd_beta,double cdd_k,double intercept, np.ndarray[np.double_t, ndim=1] T_fit_bnds, np.ndarray[np.double_t, ndim=1] T):
    
    cdef const double* T_fit_bnds_ptr = &T_fit_bnds[0]
    cdef const double* T_ptr = &T[0]
    cdef int T_size = T.size
    return full_model(hdd_bp, hdd_beta, hdd_k, cdd_bp, cdd_beta, cdd_k, intercept, T_fit_bnds_ptr, T_ptr, T_size)
    