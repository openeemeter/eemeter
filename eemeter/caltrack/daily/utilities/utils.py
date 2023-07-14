import numpy as np

from scipy.stats import t as t_dist
from scipy.stats import norm as norm_dist

from copy import deepcopy as copy

import numba
from numba.extending import overload


# To compile ahead of time: https://numba.readthedocs.io/en/stable/user/pycc.html
numba_cache = True


min_pos_system_value = (np.finfo(float).tiny * (1e20)) ** (1 / 2)
max_pos_system_value = (np.finfo(float).max * (1e-20)) ** (1 / 2)
ln_min_pos_system_value = np.log(min_pos_system_value)
ln_max_pos_system_value = np.log(max_pos_system_value)


from attrs import define
from enum import Enum

class ModelType(Enum):
    # Full model \_/
    HDD_TIDD_CDD_SMOOTH = 1
    HDD_TIDD_CDD = 2
    
    # Heating, temp independent \__
    HDD_TIDD_SMOOTH = 3
    HDD_TIDD = 4
    
    # Temp independent, cooling __/
    TIDD_CDD_SMOOTH = 5
    TIDD_CDD = 5

    # Temp independent, ___
    TIDD = 6


@define
class ModelCoefficients:
    model_type: ModelType
    intercept: float
    hdd_bp: float | None = None
    hdd_beta: float | None = None
    hdd_k: float | None = None
    cdd_bp: float | None = None
    cdd_beta: float | None = None
    cdd_k: float | None = None

    # add validator for model type and coeffs included

    @classmethod
    def from_np_arrays(cls, coefficients, coefficient_ids):
        match coefficient_ids:
            case [
                "hdd_bp",
                "hdd_beta",
                "hdd_k",
                "cdd_bp",
                "cdd_beta",
                "cdd_k",
                "intercept",
            ]:
                return ModelCoefficients(
                    model_type=ModelType.HDD_TIDD_CDD_SMOOTH,
                    hdd_bp=coefficients[0],
                    hdd_beta=coefficients[1],
                    hdd_k=coefficients[2],
                    cdd_bp=coefficients[3],
                    cdd_beta=coefficients[4],
                    cdd_k=coefficients[5],
                    intercept=coefficients[6],
                )
            case [
                "hdd_bp",
                "hdd_beta",
                "cdd_bp",
                "cdd_beta",
                "intercept",
            ]:
                return ModelCoefficients(
                    model_type=ModelType.HDD_TIDD_CDD,
                    hdd_bp=coefficients[0],
                    hdd_beta=coefficients[1],
                    cdd_bp=coefficients[2],
                    cdd_beta=coefficients[3],
                    intercept=coefficients[4],
                )
            case [
                "c_hdd_bp",
                "c_hdd_beta",
                "c_hdd_k",
                "intercept",
            ]:
                if coefficients[1] < 0: #model is heating dependent
                    hdd_bp = coefficients[0]
                    hdd_beta = -coefficients[1]
                    hdd_k = coefficients[2]
                    cdd_bp = cdd_beta = cdd_k = None
                    model_type = ModelType.HDD_TIDD_SMOOTH
                else: #model is cooling dependent
                    cdd_bp = coefficients[0]
                    cdd_beta = coefficients[1]
                    cdd_k = coefficients[2]
                    hdd_bp = hdd_beta = hdd_k = None
                    model_type = ModelType.TIDD_CDD_SMOOTH
                return ModelCoefficients(
                    model_type=model_type,
                    hdd_bp=hdd_bp,
                    hdd_beta=hdd_beta,
                    hdd_k=hdd_k,
                    cdd_bp=cdd_bp,
                    cdd_beta=cdd_beta,
                    cdd_k=cdd_k,
                    intercept=coefficients[3],
                )
            case [
                "c_hdd_bp",
                "c_hdd_beta",
                "intercept",
            ]:
                if coefficients[1] < 0: #model is heating dependent
                    hdd_bp = coefficients[0]
                    hdd_beta = -coefficients[1]
                    cdd_bp = cdd_beta = None
                    model_type = ModelType.HDD_TIDD
                else: #model is cooling dependent
                    cdd_bp = coefficients[0]
                    cdd_beta = coefficients[1]
                    hdd_bp = hdd_beta = None
                    model_type = ModelType.TIDD_CDD
                return ModelCoefficients(
                    model_type=model_type,
                    hdd_bp=hdd_bp,
                    hdd_beta=hdd_beta,
                    cdd_bp=cdd_bp,
                    cdd_beta=cdd_beta,
                    intercept=coefficients[2],
                )
            case [
                "intercept",
            ]:
                return ModelCoefficients(
                    model_type=ModelType.TIDD,
                    intercept=coefficients[0],
                )
            case _:
                raise ValueError

    def to_np_array(self):
        match self.model_type:
            case ModelType.HDD_TIDD_CDD_SMOOTH:
                return np.array([self.hdd_bp, self.hdd_beta, self.hdd_k, self.cdd_bp, self.cdd_beta, self.cdd_k, self.intercept])
            case ModelType.HDD_TIDD_CDD:
                return np.array([self.hdd_bp, self.hdd_beta, self.hdd_k, self.cdd_bp, self.cdd_beta, self.intercept])
            case ModelType.HDD_TIDD_SMOOTH:
                return np.array([self.hdd_bp, self.hdd_beta, self.hdd_k, self.intercept])
            case ModelType.TIDD_CDD_SMOOTH:
                return np.array([self.cdd_bp, self.cdd_beta, self.cdd_k, self.intercept])
            case ModelType.HDD_TIDD_SMOOTH:
                return np.array([self.hdd_bp, self.hdd_beta, self.intercept])
            case ModelType.TIDD_CDD_SMOOTH:
                return np.array([self.cdd_bp, self.cdd_beta, self.intercept])
            case ModelType.TIDD:
                return np.array([self.intercept])


from typing import Annotated
@define
class CoefficientBounds:
    model_type: ModelType
    intercept_bnds: Annotated[list[float], 2]
    hdd_bp_bnds: Annotated[list[float], 2] | None = None
    hdd_beta_bnds: Annotated[list[float], 2] | None = None
    hdd_k_bnds: Annotated[list[float], 2] | None = None
    cdd_bp_bnds: Annotated[list[float], 2] | None = None
    cdd_beta_bnds: Annotated[list[float], 2] | None = None
    cdd_k_bnds: Annotated[list[float], 2] | None = None


@overload(np.clip)
def np_clip(a, a_min, a_max):
    @numba.vectorize
    def _clip(a, a_min, a_max):
        """vectorized implementation of the clip function"""
        if np.isnan(a):
            return np.nan
        elif a < a_min:
            return a_min
        elif a > a_max:
            return a_max
        else:
            return a

    def clip_impl(a, a_min, a_max):
        """numba implementation of the clip function"""
        return _clip(a, a_min, a_max)

    return clip_impl


def OoM(x, method="round"):
    if not isinstance(x, np.ndarray):
        x = np.array(x)

    return OoM_numba(x, method=method)


@numba.jit(nopython=True, cache=numba_cache)
def OoM_numba(x, method="round"):
    x_OoM = np.empty_like(x)
    for i, xi in enumerate(x):
        if xi == 0.0:
            x_OoM[i] = 1.0

        elif method.lower() == "floor":
            x_OoM[i] = np.floor(np.log10(np.abs(xi)))

        elif method.lower() == "ceil":
            x_OoM[i] = np.ceil(np.log10(np.abs(xi)))

        elif method.lower() == "round":
            x_OoM[i] = np.round(np.log10(np.abs(xi)))

        else:  # "exact"
            x_OoM[i] = np.log10(np.abs(xi))

    return x_OoM


def RoundToSigFigs(x, p):
    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10 ** (p - 1))
    mags = 10 ** (p - 1 - OoM(x_positive))
    return np.round(x * mags) / mags


def t_stat(alpha, n, tail=2):
    degrees_of_freedom = n - 1
    if tail == "one" or tail == 1:
        perc = 1 - alpha
    elif tail == "two" or tail == 2:
        perc = 1 - alpha / 2

    return t_dist.ppf(perc, degrees_of_freedom, 0, 1)


def unc_factor(n, interval="PI", alpha=0.05):
    if interval == "CI":
        return t_stat(alpha, n) / np.sqrt(n)

    if interval == "PI":
        return t_stat(alpha, n) * (1 + 1 / np.sqrt(n))


MAD_k = 1 / norm_dist.ppf(
    0.75
)  # Conversion factor from MAD to std for normal distribution


def median_absolute_deviation(x):
    mu = np.median(x)
    sigma = np.median(np.abs(x - mu)) * MAD_k

    return sigma


@numba.jit(nopython=True, cache=numba_cache)
def weighted_std(x, w, mean=None, w_sum_err=1e-6):
    n = float(len(x))

    w_sum = np.sum(w)
    if w_sum < 1 - w_sum_err or w_sum > 1 + w_sum_err:
        w /= w_sum

    if mean is None:
        mean = np.sum(w * x)

    var = np.sum(w * np.power((x - mean), 2)) / (1 - 1 / n)

    return np.sqrt(var)


def fast_std(x, weights=None, mean=None):
    if isinstance(weights, int) or isinstance(weights, float):
        weights = np.array([weights])

    if weights is None or len(weights) == 1 or np.allclose(weights - weights[0], 0):
        if mean is None:
            return np.std(x)

        else:
            n = float(len(x))
            var = np.sum(np.power((x - mean), 2)) / n
            return np.sqrt(var)

    else:
        if mean is None:
            mean = np.average(x, weights=weights)

        return weighted_std(x, weights, mean)
