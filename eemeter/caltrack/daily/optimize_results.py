from copy import deepcopy as copy

import numpy as np

from eemeter.caltrack.daily.base_models.full_model import full_model, get_full_model_x

from eemeter.caltrack.daily.utilities.selection_criteria import selection_criteria
from eemeter.caltrack.daily.utilities.base_model import (
    get_smooth_coeffs,
    get_T_bnds,
)

from eemeter.caltrack.daily.utilities.utils import unc_factor
from eemeter.caltrack.daily.utilities.utils import ModelCoefficients


from timeit import default_timer as timer


def get_k(X, T_min_seg, T_max_seg):
    [hdd_bp, hdd_k, cdd_bp, cdd_k] = get_smooth_coeffs(*X)

    if X[0] >= T_max_seg:
        hdd_bp = X[0]
        hdd_k = 0.0

        if (cdd_k == 0) and (hdd_k == 0):
            cdd_bp = hdd_bp

    if X[2] <= T_min_seg:
        cdd_bp = X[2]
        cdd_k = 0.0

        if (cdd_k == 0) and (hdd_k == 0):
            hdd_bp = cdd_bp

    return [hdd_bp, hdd_k, cdd_bp, cdd_k]


def reduce_model(
    hdd_bp,
    hdd_beta,
    pct_hdd_k,
    cdd_bp,
    cdd_beta,
    pct_cdd_k,
    intercept,
    T_min,
    T_max,
    T_min_seg,
    T_max_seg,
    model_key,
):
    if (cdd_beta != 0) and (hdd_beta != 0) and ((pct_cdd_k != 0) or (pct_hdd_k != 0)):
        coef_id = [
            "hdd_bp",
            "hdd_beta",
            "hdd_k",
            "cdd_bp",
            "cdd_beta",
            "cdd_k",
            "intercept",
        ]
        x = [hdd_bp, hdd_beta, pct_hdd_k, cdd_bp, cdd_beta, pct_cdd_k, intercept]

        return coef_id, x

    elif (cdd_beta != 0) and (hdd_beta != 0) and (pct_cdd_k == 0) and (pct_hdd_k == 0):
        coef_id = ["hdd_bp", "hdd_beta", "cdd_bp", "cdd_beta", "intercept"]
        x = [hdd_bp, hdd_beta, cdd_bp, cdd_beta, intercept]

        return coef_id, x

    if (hdd_beta != 0) and (cdd_beta == 0) and (pct_hdd_k != 0):
        coef_id = ["c_hdd_bp", "c_hdd_beta", "c_hdd_k", "intercept"]
        if model_key == "hdd_tidd_cdd_smooth":
            [hdd_bp, hdd_k, cdd_bp, cdd_k] = get_k(
                [hdd_bp, pct_hdd_k, cdd_bp, pct_cdd_k], T_min_seg, T_max_seg
            )
            if (hdd_k == 0) and (cdd_k == 0):
                x = [hdd_bp, hdd_beta, hdd_k, cdd_bp, cdd_beta, cdd_k, intercept]

                return reduce_model(
                    *x, T_min, T_max, T_min_seg, T_max_seg, "c_hdd_tidd_smooth"
                )
        else:
            hdd_k = pct_hdd_k

        hdd_beta = -hdd_beta
        x = [hdd_bp, hdd_beta, hdd_k, intercept]

    elif (hdd_beta == 0) and (cdd_beta != 0) and (pct_cdd_k != 0):
        coef_id = ["c_hdd_bp", "c_hdd_beta", "c_hdd_k", "intercept"]
        if model_key == "hdd_tidd_cdd_smooth":
            [hdd_bp, hdd_k, cdd_bp, cdd_k] = get_k(
                [hdd_bp, pct_hdd_k, cdd_bp, pct_cdd_k], T_min_seg, T_max_seg
            )
            if (hdd_k == 0) and (cdd_k == 0):
                x = [hdd_bp, hdd_beta, hdd_k, cdd_bp, cdd_beta, cdd_k, intercept]

                return reduce_model(
                    *x, T_min, T_max, T_min_seg, T_max_seg, "c_hdd_tidd_smooth"
                )

        else:
            cdd_k = pct_cdd_k

        x = [cdd_bp, cdd_beta, cdd_k, intercept]

    elif (hdd_beta != 0) and (cdd_beta == 0) and (pct_hdd_k == 0):
        coef_id = ["c_hdd_bp", "c_hdd_beta", "intercept"]
        if hdd_bp >= T_max_seg:
            hdd_bp = T_max

        hdd_beta = -hdd_beta
        x = [hdd_bp, hdd_beta, intercept]

    elif (hdd_beta == 0) and (cdd_beta != 0) and (pct_cdd_k == 0):
        coef_id = ["c_hdd_bp", "c_hdd_beta", "intercept"]
        if cdd_bp <= T_min_seg:
            cdd_bp = T_min

        x = [cdd_bp, cdd_beta, intercept]

    elif (cdd_beta == 0) and (hdd_beta == 0):
        coef_id = ["intercept"]
        x = [intercept]

    return coef_id, x


def acf(x, lag_n=None, moving_mean_std=False):
    if lag_n is None:
        lags = range(len(x) - 1)
    else:
        lags = range(lag_n + 1)

    if moving_mean_std:
        corr = [1.0 if l == 0 else np.corrcoef(x[l:], x[:-l])[0][1] for l in lags]

        corr = np.array(corr)

    else:
        mean = x.mean()
        var = np.var(x)
        xp = x - mean
        corr = np.correlate(xp, xp, "full")[len(x) - 1 :] / var / len(x)

        corr = corr[: len(lags)]

    return corr


#consider rename
class OptimizedResult:
    def __init__(
        self,
        x,
        bnds,
        coef_id,
        loss_alpha,
        C,
        T,
        model,
        weight,
        resid,
        jac,
        mean_loss,
        TSS,
        success,
        message,
        nfev,
        time_elapsed,
        settings,
    ):
        self.coef_id = coef_id
        self.x = x
        self.num_coeffs = len(x)
        self.bnds = bnds
        #XXX maybe use model_key after it's defined below..
        self.named_coeffs = ModelCoefficients.from_np_arrays(x, coef_id)

        self.loss_alpha = loss_alpha
        self.C = C

        self.N = np.shape(T)[0]
        self.T = T
        [self.T_min, self.T_max], [self.T_min_seg, self.T_max_seg] = get_T_bnds(
            T, settings
        )

        self.obs = model - resid
        self.model = model
        self.weight = weight
        self.resid = resid
        self.wSSE = np.sum(weight * resid**2)

        self.mean_loss = mean_loss
        self.loss = mean_loss * self.N
        self.TSS = TSS

        self.settings = settings

        self.jac = []
        self.cov = []
        self.hess = []
        self.hess_inv = []
        self.x_unc = np.ones_like(x) * -1

        self._prediction_uncertainty()

        if jac is not None:  # for future uncertainty calculations
            self.jac = jac
            self.hess = jac.T * jac

            try:
                self.hess_inv = np.linalg.inv(self.hess)
            except:  # if unable to calculate inverse use Moore-Penrose pseudo-inverse
                self.hess_inv = np.linalg.pinv(self.hess)

            MSE = np.mean(resid**2)
            self.cov = MSE * self.hess_inv

            unc_alpha = self.settings.uncertainty_alpha
            self.x_unc = np.sqrt(np.diag(self.cov)) * unc_factor(
                self.DoF + 1, interval="PI", alpha=unc_alpha
            )

            print()
            print(self.jac)
            print(", ".join([f"{val:.3e}" for val in self.x]))
            print(", ".join([f"{val:.3e}" for val in self.x_unc]))
            print(f"full fcn: {self.f_unc:.2f}")
            print()

        self.success = success
        self.message = message
        self.nfev = nfev
        self.njev = -1
        self.nhev = -1
        self.nit = -1
        self.time_elapsed = time_elapsed * 1e3

        self._set_model_key()
        self._refine_model()

        self.x = np.array(self.x)

    def _prediction_uncertainty(self):  # based on std
        # residuals autocorrelation correction
        acorr = acf(
            self.resid, lag_n=1, moving_mean_std=False
        )  # only check 1 day of lag

        # using only lag-1 maybe change in the future
        lag_1 = acorr[1]
        N_eff = self.N * (1 - lag_1) / (1 + lag_1)
        self.DoF = N_eff - self.num_coeffs
        if self.DoF < 1:
            self.DoF = 1

        alpha = self.settings.uncertainty_alpha
        f_unc = np.std(self.resid) * unc_factor(
            self.DoF + 1, interval="PI", alpha=alpha
        )
        self.f_unc = f_unc

    def _set_model_key(self):
        if self.coef_id == [
            "hdd_bp",
            "hdd_beta",
            "hdd_k",
            "cdd_bp",
            "cdd_beta",
            "cdd_k",
            "intercept",
        ]:
            self.model_key = "hdd_tidd_cdd_smooth"
        elif self.coef_id == ["hdd_bp", "hdd_beta", "cdd_bp", "cdd_beta", "intercept"]:
            self.model_key = "hdd_tidd_cdd"
        elif self.coef_id == ["c_hdd_bp", "c_hdd_beta", "c_hdd_k", "intercept"]:
            self.model_key = "c_hdd_tidd_smooth"
        elif self.coef_id == ["c_hdd_bp", "c_hdd_beta", "intercept"]:
            self.model_key = "c_hdd_tidd"
        elif self.coef_id == ["intercept"]:
            self.model_key = "tidd"
        else:
            raise Exception(f"Unknown model type in 'OptimizeResult'")

        self.model_name = copy(self.model_key)
        if "c_hdd" in self.model_key:
            if self.x[self.coef_id.index("c_hdd_beta")] < 0:
                self.model_name = self.model_name.replace("c_hdd", "hdd")
            else:
                self.model_name = self.model_name.replace("c_hdd", "cdd")

    def _refine_model(self):
        # update coeffs based on model
        x = get_full_model_x(
            self.model_key,
            self.x,
            self.T_min,
            self.T_max,
            self.T_min_seg,
            self.T_max_seg,
        )

        # reduce model
        self.coef_id, self.x = reduce_model(
            *x, self.T_min, self.T_max, self.T_min_seg, self.T_max_seg, self.model_key
        )
        self.num_coeffs = len(self.x)

        self._set_model_key()

    def eval(self, T):
        x = get_full_model_x(
            self.model_key,
            self.x,
            self.T_min,
            self.T_max,
            self.T_min_seg,
            self.T_max_seg,
        )

        if self.model_key == "hdd_tidd_cdd_smooth":
            [hdd_bp, hdd_beta, pct_hdd_k, cdd_bp, cdd_beta, pct_cdd_k, intercept] = x
            [hdd_bp, hdd_k, cdd_bp, cdd_k] = get_smooth_coeffs(
                hdd_bp, pct_hdd_k, cdd_bp, pct_cdd_k
            )
            x = [hdd_bp, hdd_beta, hdd_k, cdd_bp, cdd_beta, cdd_k, intercept]

        hdd_bp, cdd_bp, intercept = x[0], x[3], x[6]
        T_fit_bnds = np.array([self.T_min, self.T_max])

        model = full_model(*x, T_fit_bnds, T)
        f_unc = np.ones_like(model) * self.f_unc

        load_only = model - intercept

        hdd_load = np.zeros_like(model)
        cdd_load = np.zeros_like(model)

        hdd_idx = np.argwhere(T <= hdd_bp).flatten()
        cdd_idx = np.argwhere(T >= cdd_bp).flatten()

        hdd_load[hdd_idx] = load_only[hdd_idx]
        cdd_load[cdd_idx] = load_only[cdd_idx]

        return model, f_unc, hdd_load, cdd_load
