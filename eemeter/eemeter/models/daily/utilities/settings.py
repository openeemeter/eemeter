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

import pydantic

from enum import Enum
from typing import Optional, Literal, Union

from eemeter.common.base_settings import BaseSettings, CustomField
import eemeter.eemeter.models.daily.utilities.const as _const
from eemeter.eemeter.models.daily.utilities.opt_settings import AlgorithmChoice


# region option definitions
class AlphaFinalType(str, Enum):
    ALL = "all"
    LAST = "last"


class ModelSelectionCriteria(str, Enum):
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


class FullModelSelection(str, Enum):
    HDD_TIDD_CDD = "hdd_tidd_cdd"
    C_HDD_TIDD = "c_hdd_tidd"
    TIDD = "tidd"

# endregion


class Season_Definition(BaseSettings):
    JANUARY: str = CustomField(default="winter")
    FEBRUARY: str = CustomField(default="winter")
    MARCH: str = CustomField(default="shoulder")
    APRIL: str = CustomField(default="shoulder")
    MAY: str = CustomField(default="shoulder")
    JUNE: str = CustomField(default="summer")
    JULY: str = CustomField(default="summer")
    AUGUST: str = CustomField(default="summer")
    SEPTEMBER: str = CustomField(default="summer")
    OCTOBER: str = CustomField(default="shoulder")
    NOVEMBER: str = CustomField(default="winter")
    DECEMBER: str = CustomField(default="winter")

    OPTIONS: list[str] = CustomField(default=["summer", "shoulder", "winter"])

    """Set dictionaries of seasons"""
    @pydantic.model_validator(mode="after")
    def set_numeric_dict(self) -> Season_Definition:
        season_dict = {}
        for month, num in _const.season_num.items():
            val = getattr(self, month.upper())
            if val not in self.OPTIONS:
                raise ValueError(f"SeasonDefinition: {val} is not a valid option. Valid options are {self.OPTIONS}")
            
            season_dict[num] = val
        
        self._MONTH_INDEX = _const.season_num
        self._NUM_DICT = season_dict
        self._ORDER = {val: i for i, val in enumerate(self.OPTIONS)}

        return self


class Weekday_Weekend_Definition(BaseSettings):
    MONDAY: str = CustomField(default="weekday")
    TUESDAY: str = CustomField(default="weekday")
    WEDNESDAY: str = CustomField(default="weekday")
    THURSDAY: str = CustomField(default="weekday")
    FRIDAY: str = CustomField(default="weekday")
    SATURDAY: str = CustomField(default="weekend")
    SUNDAY: str = CustomField(default="weekend")

    OPTIONS: list[str] = CustomField(default=["weekday", "weekend"])

    """Set dictionaries of weekday/weekend"""
    @pydantic.model_validator(mode="after")
    def set_numeric_dict(self) -> Weekday_Weekend_Definition:
        weekday_dict = {}
        for day, num in _const.weekday_num.items():
            val = getattr(self, day.upper())
            if val not in self.OPTIONS:
                raise ValueError(f"WeekdayWeekendDefinition: {val} is not a valid option. Valid options are {self.OPTIONS}")
            
            weekday_dict[num] = val
        
        self._DAY_INDEX = _const.weekday_num
        self._NUM_DICT = weekday_dict
        self._ORDER = {val: i for i, val in enumerate(self.OPTIONS)}

        return self
    

class Split_Selection_Definition(BaseSettings):
    CRITERIA: ModelSelectionCriteria = CustomField(
        default=ModelSelectionCriteria.BIC,
        developer=True,
        description="What selection criteria is used to select data splits of models",
    )

    PENALTY_MULTIPLIER: float = CustomField(
        default=0.24,
        ge=0,
        developer=True,
        description="Penalty multiplier for split selection criteria",
    )

    PENALTY_POWER: float = CustomField(
        default=2.061,
        ge=1,
        developer=True,
        description="What power should the penalty of the selection criteria be raised to",
    )

    ALLOW_SEPARATE_SUMMER: bool = CustomField(
        default=True,
        developer=True,
        description="Allow summer to be modeled separately",
    )

    ALLOW_SEPARATE_SHOULDER: bool = CustomField(
        default=True,
        developer=True,
        description="Allow shoulder to be modeled separately",
    )

    ALLOW_SEPARATE_WINTER: bool = CustomField(
        default=True,
        developer=True,
        description="Allow winter to be modeled separately",
    )

    ALLOW_SEPARATE_WEEKDAY_WEEKEND: bool = CustomField(
        default=True,
        developer=True,
        description="Allow weekdays and weekends to be modeled separately",
    )

    REDUCE_SPLITS_BY_GAUSSIAN: bool = CustomField(
        default=True,
        developer=True,
        description="Reduces splits by fitting with multivariate Gaussians and testing for overlap",
    )

    REDUCE_SPLITS_NUM_STD: list[float] = CustomField(
        default=[1.4, 0.89],
        developer=True,
        description="Number of standard deviations to use with Gaussians",
    )


def _check_developer_mode(cls):   
    for k, v in cls.model_fields.items():
        if isinstance(getattr(cls, k), BaseSettings):
            _check_developer_mode(getattr(cls, k))

        elif v.json_schema_extra["developer"] and getattr(cls, k) != v.default:
            raise ValueError(f"Developer mode is not enabled. Cannot change {k} from default value.")

    return cls


class DailySettings(BaseSettings):
    """Settings for creating the daily model.

    These settings should be converted to a dictionary before being passed to the DailyModel class.
    Be advised that any changes to the default settings deviates from OpenEEmeter standard methods and should be used with caution.

    Attributes:
        developer_mode (bool): Allows changing of developer settings
        algorithm_choice (str): Optimization algorithm choice. Developer mode only.
        initial_guess_algorithm_choice (str): Initial guess optimization algorithm choice. Developer mode only.
        full_model (str): The largest model allowed. Developer mode only.
        smoothed_model (bool): Allow smoothed models.
        allow_separate_summer (bool): Allow summer to be modeled separately.
        allow_separate_shoulder (bool): Allow shoulder to be modeled separately.
        allow_separate_winter (bool): Allow winter to be modeled separately.
        allow_separate_weekday_weekend (bool): Allow weekdays and weekends to be modeled separately.
        reduce_splits_by_gaussian (bool): Reduces splits by fitting with multivariate Gaussians and testing for overlap.
        reduce_splits_num_std (list[float]): Number of standard deviations to use with Gaussians.
        alpha_minimum (float): Alpha where adaptive robust loss function is Welsch loss.
        alpha_selection (float): Specified alpha to evaluate which is the best model type.
        alpha_final_type (str): When to use 'alpha_final: 'all': on every model, 'last': on final model, 'None': don't use.
        alpha_final (float | str | None): Specified alpha or 'adaptive' for adaptive loss in model evaluation.
        final_bounds_scalar (float | None): Scalar for calculating bounds of 'alpha_final'.
        regularization_alpha (float): Alpha for elastic net regularization.
        regularization_percent_lasso (float): Percent lasso vs (1 - perc) ridge regularization.
        segment_minimum_count (int): Minimum number of data points for HDD/CDD.
        maximum_slope_OoM_scaler (float): Scaler for initial slope to calculate bounds based on order of magnitude.
        initial_smoothing_parameter (float | None): Initial guess for the smoothing parameter.
        initial_step_percentage (float | None): Initial step-size for relevant algorithms.
        split_selection_criteria (str): What selection criteria is used to select data splits of models.
        split_selection_penalty_multiplier (float): Penalty multiplier for split selection criteria.
        split_selection_penalty_power (float): What power should the penalty of the selection criteria be raised to.
        season (Dict[int, str]): Dictionary of months and their associated season (January is 1).
        is_weekday (Dict[int, bool]): Dictionary of days (1 = Monday) and if that day is a weekday (True/False).
        uncertainty_alpha (float): Significance level used for uncertainty calculations (0 < float < 1).
        cvrmse_threshold (float): Threshold for the CVRMSE to disqualify a model.

    """

    DEVELOPER_MODE: bool = CustomField(
        default=False,
        developer=False,
        description="Developer mode flag",
    )

    SILENT_DEVELOPER_MODE: bool = CustomField(
        default=False,
        developer=False,
        exclude=True,
        repr=False,
    )

    ALGORITHM_CHOICE: Optional[AlgorithmChoice] = CustomField(
        default=AlgorithmChoice.NLOPT_SBPLX,
        developer=True,
        description="Optimization algorithm choice",
    )

    INITIAL_GUESS_ALGORITHM_CHOICE: Optional[AlgorithmChoice] = CustomField(
        default=AlgorithmChoice.NLOPT_DIRECT,
        developer=True,
        description="Initial guess optimization algorithm choice",
    )

    FULL_MODEL: Optional[FullModelSelection] = CustomField(
        default=FullModelSelection.HDD_TIDD_CDD,
        developer=True,
        description="The largest model allowed",
    )

    SMOOTHED_MODEL: bool = CustomField(
        default=True,
        developer=True,
        description="Allow smoothed models",
    )

    ALPHA_MINIMUM: float = CustomField(
        default=-100,
        le=-10,
        developer=True,
        description="Alpha where adaptive robust loss function is Welsch loss",
    )

    ALPHA_SELECTION: float = CustomField(
        default=2,
        ge=-10,
        le=2,
        developer=True,
        description="Specified alpha to evaluate which is the best model type",
    )

    ALPHA_FINAL_TYPE: Optional[AlphaFinalType] = CustomField(
        default=AlphaFinalType.LAST,
        developer=True,
        description="When to use 'alpha_final: 'all': on every model, 'last': on final model, 'None': don't use",
    )

    ALPHA_FINAL: Optional[Union[float, Literal["adaptive"]]] = CustomField(
        default="adaptive",
        developer=True,
        description="Specified alpha or 'adaptive' for adaptive loss in model evaluation",
    )

    FINAL_BOUNDS_SCALAR: Optional[float] = CustomField(
        default=1,
        developer=True,
        description="Scalar for calculating bounds of 'alpha_final'",
    )

    REGULARIZATION_ALPHA: float = CustomField(
        default=0.001,
        ge=0,
        developer=True,
        description="Alpha for elastic net regularization",
    )

    REGULARIZATION_PERCENT_LASSO: float = CustomField(
        default=1,
        ge=0,
        le=1,
        developer=True,
        description="Percent lasso vs (1 - perc) ridge regularization",
    )

    SEGMENT_MINIMUM_COUNT: int = CustomField(
        default=6,
        ge=3,
        developer=True,
        description="Minimum number of data points for HDD/CDD",
    )

    MAXIMUM_SLOPE_OOM_SCALER: float = CustomField(
        default=2,
        ge=1,
        developer=True,
        description="Scaler for initial slope to calculate bounds based on order of magnitude",
    )

    INITIAL_STEP_PERCENTAGE: Optional[float] = CustomField(
        default=0.1,
        developer=True,
        description="Initial step-size for relevant algorithms",
    )

    SPLIT_SELECTION: Split_Selection_Definition = CustomField(
        default_factory=Split_Selection_Definition,
        developer=True,
        description="Settings for split selection",
    )

    SEASON: Season_Definition = CustomField(
        default_factory=Season_Definition,
        developer=False,
        description="Dictionary of months and their associated season (January is 1)",
    )

    WEEKDAY_WEEKEND: Weekday_Weekend_Definition = CustomField(
        default_factory=Weekday_Weekend_Definition,
        developer=False,
        description="Dictionary of days (1 = Monday) and if that day is a weekday (True/False)",
    )

    UNCERTAINTY_ALPHA: float = CustomField(
        default=0.05,
        ge=0,
        le=1,
        developer=False,
        description="Significance level used for uncertainty calculations",
    )

    CVRMSE_THRESHOLD: float = CustomField(
        default=1,
        ge=0,
        developer=True,
        description="Threshold for the CVRMSE to disqualify a model",
    )


    @pydantic.model_validator(mode="after")
    def _check_developer_mode(self):
        if self.DEVELOPER_MODE:
            if not self.SILENT_DEVELOPER_MODE:
                print("Warning: Daily model is nonstandard and should be explicitly stated in any derived work")

            return self
        
        _check_developer_mode(self)

        return self


    @pydantic.model_validator(mode="after")
    def _check_alpha_final(self):
        if self.ALPHA_FINAL is None:
            if self.ALPHA_FINAL_TYPE != None:
                raise ValueError("`ALPHA_FINAL` must be set if `ALPHA_FINAL_TYPE` is not None")
            
        elif isinstance(self.ALPHA_FINAL, float):
            if (self.ALPHA_MINIMUM > self.ALPHA_FINAL) or (self.ALPHA_FINAL > 2.0):
                raise ValueError(
                    f"`ALPHA_FINAL` must be `adaptive` or `ALPHA_MINIMUM` <= float <= 2"
                )

        elif isinstance(self.ALPHA_FINAL, str):
            if self.ALPHA_FINAL != "adaptive":
                raise ValueError(
                    f"ALPHA_FINAL must be `adaptive` or `ALPHA_MINIMUM` <= float <= 2"
            )

        return self

    @pydantic.model_validator(mode="after")
    def _check_final_bounds_scalar(self):
        if self.FINAL_BOUNDS_SCALAR is not None:
            if self.FINAL_BOUNDS_SCALAR <= 0:
                raise ValueError("`FINAL_BOUNDS_SCALAR` must be > 0")
            
            if self.ALPHA_FINAL_TYPE is None:
                raise ValueError("`FINAL_BOUNDS_SCALAR` must be None if `ALPHA_FINAL` is None")
            
        else:
            if self.ALPHA_FINAL_TYPE is not None:
                raise ValueError("`FINAL_BOUNDS_SCALAR` must be > 0 if `ALPHA_FINAL` is not None")

        return self

    
    @pydantic.model_validator(mode="after")
    def _check_initial_step_percentage(self):
        if self.INITIAL_STEP_PERCENTAGE is not None:
            if self.INITIAL_STEP_PERCENTAGE <= 0 or self.INITIAL_STEP_PERCENTAGE > 0.5:
                raise ValueError("`INITIAL_STEP_PERCENTAGE` must be None or 0 < float <= 0.5")
            
        else:
            if self.ALGORITHM_CHOICE[:5] in ["nlopt"]:
                raise ValueError("`INITIAL_STEP_PERCENTAGE` must be specified if `ALGORITHM_CHOICE` is from Nlopt")
            
        return self
            
    
    def __repr__(self):
        text_all = []
        text_all.append(type(self).__name__)

        # get all keys
        keys = list(self.model_fields.keys())

        # print away
        key_max = max([len(k) for k in keys]) + 2
        for key in keys:
            if not self.model_fields[key].repr:
                continue

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


class Weekday_Weekend_Definition(BaseSettings):
    MONDAY: str = CustomField(default="weekday")
    TUESDAY: str = CustomField(default="weekday")
    WEDNESDAY: str = CustomField(default="weekday")
    THURSDAY: str = CustomField(default="weekday")
    FRIDAY: str = CustomField(default="weekday")
    SATURDAY: str = CustomField(default="weekend")
    SUNDAY: str = CustomField(default="weekend")

    OPTIONS: list[str] = CustomField(default=["weekday", "weekend"])

    """Set dictionaries of weekday/weekend"""
    @pydantic.model_validator(mode="after")
    def set_numeric_dict(self) -> Weekday_Weekend_Definition:
        weekday_dict = {}
        for day, num in _const.weekday_num.items():
            val = getattr(self, day.upper())
            if val not in self.OPTIONS:
                raise ValueError(f"WeekdayWeekendDefinition: {val} is not a valid option. Valid options are {self.OPTIONS}")
            
            weekday_dict[num] = val
        
        self._DAY_INDEX = _const.weekday_num
        self._NUM_DICT = weekday_dict
        self._ORDER = {val: i for i, val in enumerate(self.OPTIONS)}

        return self
    

class Split_Selection_Legacy_Definition(Split_Selection_Definition):
    ALLOW_SEPARATE_SUMMER: bool = CustomField(
        default=False,
        developer=True,
        description="Allow summer to be modeled separately",
    )

    ALLOW_SEPARATE_SHOULDER: bool = CustomField(
        default=False,
        developer=True,
        description="Allow shoulder to be modeled separately",
    )

    ALLOW_SEPARATE_WINTER: bool = CustomField(
        default=False,
        developer=True,
        description="Allow winter to be modeled separately",
    )

    ALLOW_SEPARATE_WEEKDAY_WEEKEND: bool = CustomField(
        default=False,
        developer=True,
        description="Allow weekdays and weekends to be modeled separately",
    )

    REDUCE_SPLITS_BY_GAUSSIAN: bool = CustomField(
        default=False,
        developer=True,
        description="Reduces splits by fitting with multivariate Gaussians and testing for overlap",
    )

    REDUCE_SPLITS_NUM_STD: Optional[list[float]] = CustomField(
        default=None,
        developer=True,
        description="Number of standard deviations to use with Gaussians",
    )


class DailyLegacySettings(DailySettings):
    SMOOTHED_MODEL: bool = CustomField(
        default=False,
        developer=True,
        description="Allow smoothed models",
    )

    ALPHA_FINAL: Optional[Union[float, Literal["adaptive"]]] = CustomField(
        default=2,
        developer=True,
        description="Specified alpha or 'adaptive' for adaptive loss in model evaluation",
    )

    SEGMENT_MINIMUM_COUNT: int = CustomField(
        default=10,
        ge=3,
        developer=True,
        description="Minimum number of data points for HDD/CDD",
    )

    SPLIT_SELECTION: Split_Selection_Legacy_Definition = CustomField(
        default_factory=Split_Selection_Legacy_Definition,
        developer=True,
        description="Settings for split selection",
    )


def update_daily_settings(settings, update_dict):
    if not isinstance(settings, DailySettings):
        raise TypeError("settings must be an instance of 'Daily_Settings'")

    # update settings with update_dict
    settings_dict = settings.model_dump()
    settings_dict.update(update_dict)

    return DailySettings(**settings_dict)


# TODO: deprecate
def default_settings(**kwargs) -> DailySettings:
    """
    Returns default settings.
    """
    return DailySettings(**kwargs)

# TODO: deprecate
def caltrack_legacy_settings(**kwargs) -> DailyLegacySettings:
    """
    Returns CalTRACK legacy settings.
    """
    return DailyLegacySettings(**kwargs)