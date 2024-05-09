#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

   Copyright 2014-2024 OpenEEmeter contributors

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

"""
from __future__ import annotations

from typing import Any, Callable, Dict

import attrs

"""
# TODO: define options

Args:
    agg_type (str, default='mean'): How to aggregate
        Options: ['mean', 'median']
"""


_KEY_DESCR = "descr"


# region option definitions
class AlgorithmChoice:
    """
    choice of optimization algorithms to use for optimization of daily models
    """

    # SciPy-based algorithms
    SCIPY_NELDERMEAD = "scipy_Nelder-Mead"
    SCIPY_L_BFGS_B = "scipy_L-BFGS-B"
    SCIPY_TNC = "scipy_TNC"
    SCIPY_SLSQP = "scipy_SLSQP"
    SCIPY_POWELL = "scipy_Powell"
    SCIPY_TRUST_CONSTR = "scipy_trust-constr"

    # nlopt-based algorithms
    NLOPT_DIRECT = "nlopt_DIRECT"
    NLOPT_DIRECT_NOSCAL = "nlopt_DIRECT_NOSCAL"
    NLOPT_DIRECT_L = "nlopt_DIRECT_L"
    NLOPT_DIRECT_L_RAND = "nlopt_DIRECT_L_RAND"
    NLOPT_DIRECT_L_NOSCAL = "nlopt_DIRECT_L_NOSCAL"
    NLOPT_DIRECT_L_RAND_NOSCAL = "nlopt_DIRECT_L_RAND_NOSCAL"
    NLOPT_ORIG_DIRECT = "nlopt_ORIG_DIRECT"
    NLOPT_ORIG_DIRECT_L = "nlopt_ORIG_DIRECT_L"
    NLOPT_CRS2_LM = "nlopt_CRS2_LM"
    NLOPT_MLSL_LDS = "nlopt_MLSL_LDS"
    NLOPT_MLSL = "nlopt_MLSL"
    NLOPT_STOGO = "nlopt_STOGO"
    NLOPT_STOGO_RAND = "nlopt_STOGO_RAND"
    NLOPT_AGS = "nlopt_AGS"
    NLOPT_ISRES = "nlopt_ISRES"
    NLOPT_ESCH = "nlopt_ESCH"
    NLOPT_COBYLA = "nlopt_COBYLA"
    NLOPT_BOBYQA = "nlopt_BOBYQA"
    NLOPT_NEWUOA = "nlopt_NEWUOA"
    NLOPT_NEWUOA_BOUND = "nlopt_NEWUOA_BOUND"
    NLOPT_PRAXIS = "nlopt_PRAXIS"
    NLOPT_NELDERMEAD = "nlopt_NELDERMEAD"
    NLOPT_SBPLX = "nlopt_SBPLX"
    NLOPT_MMA = "nlopt_MMA"
    NLOPT_CCSAQ = "nlopt_CCSAQ"
    NLOPT_SLSQP = "nlopt_SLSQP"
    NLOPT_L_BFGS = "nlopt_LBFGS"
    NLOPT_TNEWTON = "nlopt_TNEWTON"
    NLOPT_TNEWTON_PRECOND = "nlopt_TNEWTON_PRECOND"
    NLOPT_TNEWTON_RESTART = "nlopt_TNEWTON_RESTART"
    NLOPT_TNEWTON_PRECOND_RESTART = "nlopt_TNEWTON_PRECOND_RESTART"
    NLOPT_VAR1 = "nlopt_VAR1"
    NLOPT_VAR2 = "nlopt_VAR2"


class AlphaFinalType:
    ALL = "all"
    LAST = "last"
    NONE = None


class ModelSelectionCriteria:
    RMSE = "rmse"
    RMSE_ADJ = "rmse_adj"
    R_SQUARED = "r_squared"
    R_SQUARED_ADJ = "r_squared_adj"
    AIC = "aic"
    AICC = "aicc"
    CAIC = "caic"
    BIC = "bic"
    SABIC = "sabic"
    FPE = "fpe"

    # Maybe these will be implemented one day
    # DIC = "dic"
    # WAIC = "waic"
    # WBIC = "wbic"


class FullModelSelection:
    HDD_TIDD_CDD = "hdd_tidd_cdd"
    C_HDD_TIDD = "c_hdd_tidd"
    TIDD = "tidd"


# endregion


# region private
def _get_pub_class_attrib_dict(cls: type):
    return {k: v for k, v in cls.__dict__.items() if not k.startswith("_")}


def get_pub_class_attrib_values(cls: type):
    """
    Returns a tuple of all the public class attribute values.

    Helpful for returning a list of enum values type to a class.

    ex.
    class MatchType:
        DISTANCE_MATCH = "distance_match"
        STRATIFIED_SAMPLE = "stratified_sample"

    get_pub_class_attrib_values(MatchType) -> ("distance_match", "stratified_sample")
    """
    return tuple(_get_pub_class_attrib_dict(cls).values())


# endregion

full_algo_list = [
    getattr(AlgorithmChoice, key).lower()
    for key in dir(AlgorithmChoice)
    if "__" != key[0:2]
]


def _algorithm_white_list(
    white_list: list[str] | None = None, black_list: list[str] | None = None
):
    if white_list is not None and len(white_list) == 0:
        white_list = None

    if black_list is not None and len(black_list) == 0:
        black_list = None

    if white_list is None and black_list is None:
        return full_algo_list

    if white_list is not None:
        return white_list

    elif black_list is not None:
        return [algo for algo in full_algo_list if algo not in black_list]


# region validation
def developer_mode_validation(callable: Callable[[Any], bool]):
    def inner(instance, attribute, v):
        if (v != attribute.default) and not instance.developer_mode:
            text = [
                f"'{attribute.name}' can only be changed if 'developer_mode' == True.",
                "Warning: This is nonstandard and should be explicitly stated in any derived work",
            ]
            raise ValueError("\n".join(text))

        callable(instance, attribute, v)

    return inner


def simple_validation(
    callable: Callable[[Any], bool],
    err_msg: str = "",
    algorithm_white_list: list[str] = full_algo_list,
    dev_setting: bool = False,
):
    """
    provides a simple way to provide inline validation using a lambda predicate.
    PREDICATE RETURN TRUE FOR SUCCESS
    Helpful because validators must take three inputs.
    Returns a function which provides the necessary function structure but only runs the provided predicate inside
    """

    def inner(instance, attribute, v):
        if dev_setting and (v != attribute.default) and not instance.developer_mode:
            text = [
                f"'{attribute.name}' can only be changed if 'developer_mode' == True.",
                "Warning: This is nonstandard and should be explicitly stated in any derived work",
            ]
            raise ValueError("\n".join(text))

        if v is None and instance.algorithm_choice in algorithm_white_list:
            raise ValueError(
                f"{attribute.name} must be defined for the '{instance.algorithm_choice}' algorithm"
            )

        elif v is not None and instance.algorithm_choice not in algorithm_white_list:
            raise ValueError(
                f"{attribute.name} must be None for the '{instance.algorithm_choice}' algorithm"
            )

        elif not callable(v):
            raise ValueError(f"{err_msg} (Input value: {v})")

    return inner


def algorithm_choice_validator(instance: DailySettings, attribute: str, value: str):
    if value not in [
        s.lower() for s in _get_pub_class_attrib_dict(AlgorithmChoice).values()
    ]:
        raise ValueError(f"invalid selection for {attribute.name}")


def full_model_validator(instance: DailySettings, attribute: str, value: str):
    if value not in [
        s.lower() for s in _get_pub_class_attrib_dict(FullModelSelection).values()
    ]:
        raise ValueError(f"invalid selection for {attribute.name}")


def alpha_final_validator(
    instance: DailySettings, attribute: str, value: float | str | None
):
    if value is None:
        if instance.alpha_final_type != None:
            raise ValueError(
                f"{attribute.name} must be 'adaptive' or 'alpha_minimum' <= float <= 2 if 'alpha_final_type' is {instance.alpha_final_type}"
            )

    elif isinstance(value, float):
        if (instance.alpha_minimum > value) or (value > 2.0):
            raise ValueError(
                f"{attribute.name} must be 'adaptive' or 'alpha_minimum' <= float <= 2"
            )

    elif isinstance(value, str):
        if value != "adaptive":
            raise ValueError(
                f"{attribute.name} must be 'adaptive' or 'alpha_minimum' <= float <= 2"
            )


def final_bounds_scalar_validator(
    instance: DailySettings, attribute: str, value: float | None
):
    if value is not None:
        if value < 0.0:
            raise ValueError(f"{attribute.name} must be None or 0 < float")

        if instance.alpha_final_type is None:
            raise ValueError(
                f"{attribute.name} must be None if 'alpha_final_type' is None"
            )

    else:
        if instance.alpha_final_type is not None:
            raise ValueError(
                f"{attribute.name} must be 0 < float if 'alpha_final_type' is None"
            )


def initial_smoothing_parameter_validator(
    instance: DailySettings, attribute: str, value: float | None
):
    if value is not None:
        if value < 0.0:
            raise ValueError(f"{attribute.name} must be None or 0 < float")

    else:
        if instance.include_new_base_models:
            raise ValueError(
                f"{attribute.name} must be specified if 'include_new_base_models' is True"
            )


def initial_step_percentage_validator(
    instance: DailySettings, attribute: str, value: float | None
):
    if value is not None:
        if value <= 0.0 or value > 0.5:
            raise ValueError(f"{attribute.name} must be None or 0 < float <= 0.5")

    else:
        if instance.AlgorithmChoice[:5] in ["nlopt"]:
            raise ValueError(
                f"{attribute.name} must be specified if 'algorithm_choice' is from Nlopt"
            )


def season_choice_validator(
    instance: DailySettings, attribute: str, season_dict: Dict[int, str]
):
    min_cnt = 1

    cnt = {"summer": 0, "shoulder": 0, "winter": 0}
    for key, value in season_dict.items():
        if (key < 1) or (key > 12):
            raise ValueError(
                f"all month values must be 1 <= x <= 12 selection for {attribute.name}"
            )

        if value not in ["summer", "shoulder", "winter"]:
            raise ValueError(
                f"season value must be ['summer', 'shoulder', 'winter'] for {attribute.name}"
            )

        cnt[value] += 1

    for key, value in cnt.items():
        if value < min_cnt:
            raise ValueError(
                f"{key} must be assigned to at least {min_cnt} month for {attribute.name}"
            )


def is_weekday_validator(
    instance: DailySettings, attribute: str, is_weekday_dict: Dict[int, bool]
):
    weekend_cnt = 0
    for key, value in is_weekday_dict.items():
        if (key < 1) or (key > 7):
            raise ValueError(
                f"all day values must be 1 <= x <= 7 (1 is Monday) for {attribute.name}"
            )

        if not isinstance(value, bool):
            raise ValueError(f"is_weekday value must be boolean for {attribute.name}")

        if not value:
            weekend_cnt += 1

    if weekend_cnt != 2:
        raise ValueError(f"There must be 2 weekend days specified for {attribute.name}")


# endregion


@attrs.define(kw_only=True)
class DailySettings:
    """
    Create settings for calculating the daily model
    """

    developer_mode: bool = attrs.field(
        validator=simple_validation(
            lambda x: isinstance(x, bool),
        ),
        metadata={_KEY_DESCR: "allows changing of developer settings"},
        default=False,
    )

    algorithm_choice: str = attrs.field(
        converter=lambda x: x.lower() if isinstance(x, str) else x,
        validator=developer_mode_validation(algorithm_choice_validator),
        metadata={_KEY_DESCR: "optimization algorithm choice"},
        on_setattr=attrs.setters.frozen,
        default=AlgorithmChoice.NLOPT_SBPLX.lower(),
    )

    initial_guess_algorithm_choice: str = attrs.field(
        converter=lambda x: x.lower() if isinstance(x, str) else x,
        validator=developer_mode_validation(algorithm_choice_validator),
        metadata={_KEY_DESCR: "initial guess optimization algorithm choice"},
        on_setattr=attrs.setters.frozen,
        default=AlgorithmChoice.NLOPT_DIRECT.lower(),  # AlgorithmChoice.NLOPT_STOGO
    )

    full_model: str = attrs.field(
        converter=lambda x: x.lower() if isinstance(x, str) else x,
        validator=developer_mode_validation(full_model_validator),
        metadata={_KEY_DESCR: "the largest model allowed"},
        on_setattr=attrs.setters.frozen,
        default=FullModelSelection.HDD_TIDD_CDD,
    )

    smoothed_model: bool = attrs.field(
        validator=simple_validation(
            lambda x: isinstance(x, bool),
            dev_setting=True,
        ),
        metadata={_KEY_DESCR: "allow smoothed models"},
        on_setattr=attrs.setters.frozen,
        default=True,
    )

    allow_separate_summer: bool = attrs.field(
        validator=simple_validation(
            lambda x: isinstance(x, bool),
            dev_setting=True,
        ),
        metadata={_KEY_DESCR: "allow summer to be modeled separately"},
        on_setattr=attrs.setters.frozen,
        default=True,
    )

    allow_separate_shoulder: bool = attrs.field(
        validator=simple_validation(
            lambda x: isinstance(x, bool),
            dev_setting=True,
        ),
        metadata={_KEY_DESCR: "allow shoulder to be modeled separately"},
        on_setattr=attrs.setters.frozen,
        default=True,
    )

    allow_separate_winter: bool = attrs.field(
        validator=simple_validation(
            lambda x: isinstance(x, bool),
            dev_setting=True,
        ),
        metadata={_KEY_DESCR: "allow winter to be modeled separately"},
        on_setattr=attrs.setters.frozen,
        default=True,
    )

    allow_separate_weekday_weekend: bool = attrs.field(
        validator=simple_validation(
            lambda x: isinstance(x, bool),
            dev_setting=True,
        ),
        metadata={_KEY_DESCR: "allow weekdays and weekends to be modeled separately"},
        on_setattr=attrs.setters.frozen,
        default=True,
    )

    reduce_splits_by_gaussian: bool = attrs.field(
        validator=simple_validation(
            lambda x: isinstance(x, bool),
            dev_setting=True,
        ),
        metadata={
            _KEY_DESCR: "reduces splits by fitting with multivariate Gaussians and testing for overlap"
        },
        on_setattr=attrs.setters.frozen,
        default=True,
    )

    # TODO: Fix the conversion and validator
    reduce_splits_num_std: list[float] = attrs.field(
        # converter=lambda x: float(x) if isinstance(x, int) else x,
        # validator=simple_validation(
        #     lambda x: isinstance(x, float) and (0 < x), "must be 0 < float"
        #     dev_setting = True,
        # ),
        metadata={_KEY_DESCR: "number of standard deviations to use with Gaussians"},
        on_setattr=attrs.setters.frozen,
        default=[1.4, 0.89],
    )

    alpha_minimum: float = attrs.field(
        converter=lambda x: float(x) if isinstance(x, int) else x,
        validator=simple_validation(
            lambda x: isinstance(x, float) and (x <= -10),
            "'alpha_minimum', must be float < -10",
            dev_setting=True,
        ),
        metadata={
            _KEY_DESCR: "alpha where adaptive robust loss function is Welsch loss"
        },
        on_setattr=attrs.setters.frozen,
        default=-100,
    )

    alpha_selection: float = attrs.field(
        converter=lambda x: float(x) if isinstance(x, int) else x,
        validator=simple_validation(
            lambda x: isinstance(x, float) and (-10 <= x) and (x <= 2),
            "'alpha_selection' must be -10 <= float <= 2",
            dev_setting=True,
        ),
        metadata={
            _KEY_DESCR: "specified alpha to evaluate which is the best model type"
        },
        on_setattr=attrs.setters.frozen,
        default=2,
    )

    alpha_final_type: str = attrs.field(
        converter=lambda x: x.lower() if isinstance(x, str) else x,
        validator=developer_mode_validation(
            attrs.validators.in_(get_pub_class_attrib_values(AlphaFinalType))
        ),
        metadata={
            _KEY_DESCR: "when to use 'alpha_final: 'all': on every model, 'last': on final model, None: don't use"
        },
        on_setattr=attrs.setters.frozen,
        default="last",
    )

    alpha_final: float | str | None = attrs.field(
        converter=lambda x: float(x) if isinstance(x, int) else x,
        validator=developer_mode_validation(alpha_final_validator),
        metadata={
            _KEY_DESCR: "specified alpha or 'adaptive' for adaptive loss in model evaluation"
        },
        on_setattr=attrs.setters.frozen,
        default="adaptive",
    )

    final_bounds_scalar: float | None = attrs.field(
        converter=lambda x: float(x) if isinstance(x, int) else x,
        validator=developer_mode_validation(final_bounds_scalar_validator),
        metadata={_KEY_DESCR: "scalar for calculating bounds of 'alpha_final'"},
        on_setattr=attrs.setters.frozen,
        default=1,
    )

    regularization_alpha: float = attrs.field(
        converter=lambda x: float(x) if isinstance(x, int) else x,
        validator=simple_validation(
            lambda x: 0 <= x,
            "'regularization_alpha' must be 0 <= float",
            dev_setting=True,
        ),
        metadata={_KEY_DESCR: "alpha for elastic net regularization"},
        on_setattr=attrs.setters.frozen,
        default=0.001,
    )

    regularization_percent_lasso: float = attrs.field(
        converter=lambda x: float(x) if isinstance(x, int) else x,
        validator=simple_validation(
            lambda x: (0 <= x) and (x <= 1),
            "'regularization_percent_lasso' must be 0 <= float <= 1",
            dev_setting=True,
        ),
        metadata={_KEY_DESCR: "percent lasso vs (1 - perc) ridge regularization"},
        on_setattr=attrs.setters.frozen,
        default=1,
    )

    segment_minimum_count: int = attrs.field(
        converter=lambda x: int(x) if isinstance(x, float) else x,
        validator=simple_validation(
            lambda x: isinstance(x, int) and (3 <= x),
            "'segment_minimum_count' must be 3 <= int",
            dev_setting=True,
        ),
        metadata={_KEY_DESCR: "minimum number of data points for HDD/CDD"},
        on_setattr=attrs.setters.frozen,
        default=6,
    )

    maximum_slope_OoM_scaler: float = attrs.field(
        converter=lambda x: float(x) if isinstance(x, int) else x,
        validator=simple_validation(
            lambda x: isinstance(x, float) and (0 < x),
            "'maximum_slope_OoM_scaler' must be 0 < float",
            dev_setting=True,
        ),
        metadata={
            _KEY_DESCR: "scaler for initial slope to calculate bounds based on order of magnitude"
        },
        on_setattr=attrs.setters.frozen,
        default=2,
    )

    initial_smoothing_parameter: float | None = attrs.field(
        converter=lambda x: float(x) if isinstance(x, int) else x,
        validator=developer_mode_validation(initial_smoothing_parameter_validator),
        metadata={_KEY_DESCR: "initial guess for the smoothing parameter"},
        on_setattr=attrs.setters.frozen,
        default=0.5,
    )

    initial_step_percentage: float | None = attrs.field(
        converter=lambda x: float(x) if isinstance(x, int) else x,
        validator=developer_mode_validation(initial_step_percentage_validator),
        metadata={_KEY_DESCR: "initial step-size for relevant algorithms"},
        on_setattr=attrs.setters.frozen,
        default=0.10,
    )

    split_selection_criteria: str = attrs.field(
        converter=lambda x: x.lower() if isinstance(x, str) else x,
        validator=attrs.validators.in_(
            get_pub_class_attrib_values(ModelSelectionCriteria)
        ),
        metadata={
            _KEY_DESCR: "what selection criteria is used to select data splits of models"
        },
        on_setattr=attrs.setters.frozen,
        default="bic",
    )

    # TODO: If this is permanent then it needs to be set up to work for only valid split_selection_criteria
    split_selection_penalty_multiplier: float = attrs.field(
        converter=lambda x: float(x) if isinstance(x, int) else x,
        validator=simple_validation(
            lambda x: isinstance(x, float) and (0 <= x),
            "must be 0 <= float",
            dev_setting=True,
        ),
        metadata={
            _KEY_DESCR: "what multiplier should be applied to the penalty of the selection criteria"
        },
        on_setattr=attrs.setters.frozen,
        default=0.240,
    )

    # TODO: If this is permanent then it needs to be set up to work for only valid split_selection_criteria
    split_selection_penalty_power: float = attrs.field(
        converter=lambda x: float(x) if isinstance(x, int) else x,
        validator=simple_validation(
            lambda x: isinstance(x, float) and (0 <= x),
            "must be 0 <= float",
            dev_setting=True,
        ),
        metadata={
            _KEY_DESCR: "what power should the penalty of the selection criteria be raised to"
        },
        on_setattr=attrs.setters.frozen,
        default=2.061,
    )

    season: Dict[int, str] = attrs.field(
        converter=lambda x: {int(k): str(v).lower().strip() for k, v in x.items()},
        validator=season_choice_validator,
        metadata={
            _KEY_DESCR: "dictionary of months and their associated season (January is 1)"
        },
        on_setattr=attrs.setters.frozen,
        default={
            1: "winter",
            2: "winter",
            3: "shoulder",
            4: "shoulder",
            5: "shoulder",
            6: "summer",
            7: "summer",
            8: "summer",
            9: "summer",
            10: "shoulder",
            11: "winter",
            12: "winter",
        },
    )

    is_weekday: Dict[int, bool] = attrs.field(
        validator=is_weekday_validator,
        converter=lambda x: {int(k): v for k, v in x.items()},
        metadata={
            _KEY_DESCR: "dictionary of days (1 = Monday) and if that day is a weekday (True/False)"
        },
        on_setattr=attrs.setters.frozen,
        default={
            1: True,
            2: True,
            3: True,
            4: True,
            5: True,
            6: False,
            7: False,
        },
    )

    uncertainty_alpha: float = attrs.field(
        converter=lambda x: float(x) if isinstance(x, int) else x,
        validator=simple_validation(
            lambda x: isinstance(x, float) and ((0 < x) and (x < 1)),
            "'uncertainty_alpha' must be 0 < float < 1",
            dev_setting=False,
        ),
        metadata={
            _KEY_DESCR: "significance level used for uncertainty calculations (0 < float < 1)"
        },
        on_setattr=attrs.setters.frozen,
        default=0.1,
    )

    cvrmse_threshold: float = attrs.field(
        converter=lambda x: float(x) if isinstance(x, int) else x,
        validator=simple_validation(
            lambda x: isinstance(x, float) and (0 < x),
            "'cvrmse_threshold' must be 0 < float",
            dev_setting=False,
        ),
        metadata={_KEY_DESCR: "threshold for the CVRMSE to disqualify a model"},
        on_setattr=attrs.setters.frozen,
        default=1.0,
    )

    def to_dict(self):
        keys = []
        config = {}
        for key in dir(self):
            if not key.startswith("_") and key != "to_dict":
                keys.append(key)
        for key in keys:
            config[key] = getattr(self, key)
        return config

    def __repr__(self):
        text_all = []
        text_all.append(type(self).__name__)

        # get all keys
        keys = []
        for key in dir(self):
            if "__" != key[0:2] and key != "to_dict":
                keys.append(key)

        # print away
        key_max = max([len(k) for k in keys]) + 2
        for key in keys:
            val = getattr(self, key)

            if isinstance(val, dict):
                v_max = max([len(str(v)) for v in list(val.values())])
                k_max = max([len(str(k)) for k in list(val.keys())])
                if k_max == 1:
                    k_max = 2

                for n, (k, v) in enumerate(val.items()):
                    if n == 0:
                        text_all.append(f"{key:>{key_max}s}: {{{str(k):>{k_max}s}: {v}")

                    elif n < len(val) - 1:
                        text_all.append(f"{'':>{key_max}s}   {str(k):>{k_max}s}: {v}")

                    else:
                        text_all.append(
                            f"{'':>{key_max}s}   {str(k):>{k_max}s}: {str(v):{v_max}s} }}"
                        )

            else:
                if isinstance(val, str):
                    val = f"'{val}'"

                text_all.append(f"{key:>{key_max}s}: {val}")

        return "\n".join(text_all)


# TODO: Currently update_daily_settings only works for 4.0. If updating 2.0 it will fail
#       This could be fixed by adding a setting to indicate 2.0 or 4.0 and then set defaults
#       based on that setting. Maybe using 'default = attrs.Factory(default_fcn)'
def update_daily_settings(settings, update_dict):
    if not isinstance(settings, DailySettings):
        raise TypeError("settings must be an instance of 'Daily_Settings'")

    return attrs.evolve(settings, **update_dict)


def default_settings(**kwargs) -> DailySettings:
    """
    Returns default settings.
    """

    settings = DailySettings(**kwargs)

    return settings


def caltrack_legacy_settings(**kwargs) -> DailySettings:
    settings = {
        "developer_mode": True,
        "algorithm_choice": "nlopt_SBPLX",  # "scipy_SLSQP",
        "initial_guess_algorithm_choice": "nlopt_DIRECT",
        "alpha_selection": 2.0,
        "alpha_final": 2.0,
        "alpha_final_type": "last",
        "regularization_alpha": 0.001,
        "regularization_percent_lasso": 1.0,
        "smoothed_model": False,
        "allow_separate_summer": False,
        "allow_separate_shoulder": False,
        "allow_separate_winter": False,
        "allow_separate_weekday_weekend": False,
        "reduce_splits_by_gaussian": False,
        "segment_minimum_count": 10,
    }

    # Check development attributes vs kwargs
    default_settings = DailySettings()

    dev_mode = "developer_mode" in kwargs and kwargs["developer_mode"]
    non_dev_settings = ["developer_mode", "season", "is_weekday", "uncertainty_alpha"]
    for key, val in kwargs.items():
        if key in settings:
            default = settings[key]
        else:
            default = getattr(default_settings, key)

        if (val != default) and (key not in non_dev_settings) and (not dev_mode):
            text = [
                f"'{key}' can only be changed if 'developer_mode' == True.",
                "Warning: This is nonstandard and should be explicitly stated in any derived work",
            ]
            raise ValueError("\n".join(text))

    # Set daily settings
    settings.update(kwargs)

    settings = DailySettings(**settings)
    settings.developer_mode = False

    return settings


if __name__ == "__main__":
    print("Default Settings")
    # print(caltrack_2_1_settings())
    # print(caltrack_legacy_settings())
    test = caltrack_legacy_settings()

    print(update_daily_settings(test, {"regularization_alpha": 0.01}))
