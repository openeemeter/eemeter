import numpy as np
from copy import deepcopy as copy

from eemeter.caltrack.daily.base_models.tidd import fit_tidd
from eemeter.caltrack.daily.base_models.c_hdd_tidd import fit_c_hdd_tidd
from eemeter.caltrack.daily.base_models.c_hdd_tidd_smooth import fit_c_hdd_tidd_smooth
from eemeter.caltrack.daily.base_models.hdd_tidd_cdd import fit_hdd_tidd_cdd
from eemeter.caltrack.daily.base_models.hdd_tidd_cdd_smooth import (
    fit_hdd_tidd_cdd_smooth,
)

from eemeter.caltrack.daily.utilities.utils import OoM, ModelCoefficients, ModelType
from eemeter.caltrack.daily.utilities.config import FullModelSelection

from eemeter.caltrack.daily.optimize_results import OptimizedResult


def _get_opt_options(settings):
    # TODO: opt_options can be removed in place of settings in the future
    opt_options = {
        "global": {
            "algorithm": settings.algorithm_choice,
            "stop_criteria_type": "Iteration Maximum",
            "stop_criteria_val": 2000,
            "initial_step": settings.initial_step_percentage,
            "xtol_rel": 1e-5,
            "ftol_rel": 1e-5,
            "initial_pop_multiplier": 2,
        },
        "local": {},
    }

    return opt_options


def fit_initial_models_from_full_model(df_meter, settings, print_res=False):
    T = df_meter["temperature"].values
    obs = df_meter["observed"].values

    opt_options = _get_opt_options(settings)
    fit_input = [T, obs, settings, opt_options]

    # initial fitting of the most complicated model allowed
    if (
        settings.full_model == FullModelSelection.HDD_TIDD_CDD
    ) and settings.smoothed_model:
        model_res = fit_hdd_tidd_cdd_smooth(*fit_input, initial_fit=True)

    elif settings.full_model == FullModelSelection.HDD_TIDD_CDD:
        model_res = fit_hdd_tidd_cdd(*fit_input, initial_fit=True)

    elif (
        settings.full_model == FullModelSelection.C_HDD_TIDD
    ) and settings.smoothed_model:
        model_res = fit_c_hdd_tidd_smooth(*fit_input, initial_fit=True)

    elif settings.full_model == FullModelSelection.C_HDD_TIDD:
        model_res = fit_c_hdd_tidd(*fit_input, initial_fit=True)

    elif settings.full_model == FullModelSelection.TIDD:
        model_res = fit_tidd(*fit_input, initial_fit=True)

    if print_res:
        criterion = model_res.selection_criterion
        # print(f"{model_key:<30s} {model_res.loss:<8.3f} {model_res.alpha:<8.2f} {model_res.C:<8.3f} {model_res.time_elapsed:>8.2f} ms")
        print(
            f"{model_res.model_name:<30s} {criterion:<8.3g} {model_res.alpha:<8.2f} {model_res.time_elapsed:>8.2f} ms"
        )

    return model_res


def fit_model(model_key, fit_input, x0: ModelCoefficients, bnds):
    if model_key == "hdd_tidd_cdd_smooth":
        res = fit_hdd_tidd_cdd_smooth(*fit_input, x0, bnds, initial_fit=False)

    elif model_key == "hdd_tidd_cdd":
        res = fit_hdd_tidd_cdd(*fit_input, x0, bnds, initial_fit=False)

    elif model_key == "c_hdd_tidd_smooth":
        res = fit_c_hdd_tidd_smooth(*fit_input, x0, bnds, initial_fit=False)

    elif model_key == "c_hdd_tidd":
        # temporary fix prior to implementing named coefficients for unsmoothed model
        if x0.model_type == ModelType.HDD_TIDD_SMOOTH:
            x0.model_type = ModelType.HDD_TIDD
        if x0.model_type == ModelType.TIDD_CDD_SMOOTH:
            x0.model_type = ModelType.TIDD_CDD
        res = fit_c_hdd_tidd(*fit_input, x0.to_np_array(), bnds, initial_fit=False)

    elif model_key == "tidd":
        res = fit_tidd(*fit_input, x0, bnds, initial_fit=False)

    return res


def fit_final_model(df_meter, HoF: OptimizedResult, settings, print_res=False):
    def get_bnds(x0, bnds_scalar):
        x_oom = 10 ** (OoM(x0, method="exact") + np.log10(bnds_scalar))
        bnds = (x0 + (np.array([-1, 1]) * x_oom[:, None]).T).T

        return bnds

    T = df_meter["temperature"].values
    obs = df_meter["observed"].values

    opt_options = _get_opt_options(settings)
    fit_input = [T, obs, settings, opt_options]

    x0 = HoF.x
    bnds = get_bnds(x0, settings.final_bounds_scalar)

    HoF = fit_model(HoF.model_key, fit_input, HoF.named_coeffs, bnds)

    if print_res:
        print(
            f"{HoF.model_name:<30s} {HoF.loss_alpha:<8.2f} {HoF.time_elapsed:>8.2f} ms"
        )

    return HoF
