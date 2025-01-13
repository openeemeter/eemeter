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
import pytest

from eemeter.eemeter.models.daily.utilities.config import (
    DailySettings,
)


def test_default_settings():
    settings = DailySettings()
    assert settings.DEVELOPER_MODE is False
    assert settings.algorithm_choice.lower() == "nlopt_sbplx"
    assert settings.initial_guess_algorithm_choice.lower() == "nlopt_direct"
    assert settings.alpha_selection == 2.0
    assert settings.alpha_final == "adaptive"
    assert settings.alpha_final_type == "last"
    assert settings.regularization_alpha == 0.001
    assert settings.regularization_percent_lasso == 1.0
    assert settings.SMOOTHED_MODEL is True
    assert settings.ALLOW_SEPARATE_SUMMER is True
    assert settings.ALLOW_SEPARATE_SHOULDER is True
    assert settings.ALLOW_SEPARATE_WINTER is True
    assert settings.ALLOW_SEPARATE_WEEKDAY_WEEKEND is True
    assert settings.REDUCE_SPLITS_BY_GAUSSIAN is True
    assert settings.segment_minimum_count == 6


def test_custom_settings():
    settings = DailySettings(
        DEVELOPER_MODE=True,
        algorithm_choice="scipy_SLSQP",
        initial_guess_algorithm_choice="nlopt_DIRECT_L",
        alpha_selection=1.5,
        alpha_final=1.5,
        alpha_final_type="last",
        regularization_alpha=0.01,
        regularization_percent_lasso=0.5,
        SMOOTHED_MODEL=True,
        ALLOW_SEPARATE_SUMMER=True,
        ALLOW_SEPARATE_SHOULDER=True,
        ALLOW_SEPARATE_WINTER=True,
        ALLOW_SEPARATE_WEEKDAY_WEEKEND=True,
        REDUCE_SPLITS_BY_GAUSSIAN=True,
        segment_minimum_count=20,
    )
    assert settings.DEVELOPER_MODE is True
    assert settings.algorithm_choice.lower() == "scipy_slsqp"
    assert settings.initial_guess_algorithm_choice.lower() == "nlopt_direct_l"
    assert settings.alpha_selection == 1.5
    assert settings.alpha_final == 1.5
    assert settings.alpha_final_type == "last"
    assert settings.regularization_alpha == 0.01
    assert settings.regularization_percent_lasso == 0.5
    assert settings.SMOOTHED_MODEL is True
    assert settings.ALLOW_SEPARATE_SUMMER is True
    assert settings.ALLOW_SEPARATE_SHOULDER is True
    assert settings.ALLOW_SEPARATE_WINTER is True
    assert settings.ALLOW_SEPARATE_WEEKDAY_WEEKEND is True
    assert settings.REDUCE_SPLITS_BY_GAUSSIAN is True
    assert settings.segment_minimum_count == 20


def test_invalid_settings():
    with pytest.raises(TypeError):
        DailySettings(DEVELOPER_MODE=False, invalid_key="invalid_value")
    with pytest.raises(ValueError):
        DailySettings(DEVELOPER_MODE=False, algorithm_choice="invalid_algorithm")
    with pytest.raises(ValueError):
        DailySettings(DEVELOPER_MODE=False, alpha_selection=0.5)
    with pytest.raises(ValueError):
        DailySettings(DEVELOPER_MODE=False, alpha_selection=1.5)
    with pytest.raises(ValueError):
        DailySettings(DEVELOPER_MODE=False, alpha_final_type="invalid_type")
