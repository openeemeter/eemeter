import numpy as np
from eemeter.caltrack.daily.base_models.full_model_import_finder import full_model


def test_full_model_import():
    hdd_bp = 50
    hdd_beta = 0.01
    hdd_k = 0.001
    cdd_bp = 80
    cdd_beta = 0.02
    cdd_k = 0.002
    intercept = 100
    T_fit_bnds = np.array([10, 100]).astype(np.double)
    T = np.linspace(10, 100, 130).astype(np.double)
    
    res = full_model(hdd_bp,hdd_beta, hdd_k, cdd_bp, cdd_beta, cdd_k, intercept, T_fit_bnds, T)
    assert res.size == T.size