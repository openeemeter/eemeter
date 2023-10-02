#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

   Copyright 2014-2023 OpenEEmeter contributors

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
import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from eemeter.visualization import plot_energy_signature, plot_time_series
from eemeter.models import DailyModel


def test_plot_time_series(il_electricity_cdd_hdd_daily):
    meter_data = il_electricity_cdd_hdd_daily["meter_data"]
    temperature_data = il_electricity_cdd_hdd_daily["temperature_data"]
    ax_m, ax_t = plot_time_series(meter_data, temperature_data)
    m_data = ax_m.lines[0].get_xydata()
    t_data = ax_t.lines[0].get_xydata()
    assert m_data.shape == (810, 2)
    assert t_data.shape == (19417, 2)


def test_plot_energy_signature(il_electricity_cdd_hdd_daily):
    meter_data = il_electricity_cdd_hdd_daily["meter_data"]
    temperature_data = il_electricity_cdd_hdd_daily["temperature_data"]
    ax = plot_energy_signature(meter_data, temperature_data, title="title")
    data = ax.collections[0].get_offsets()
    assert data.shape == (810, 2)
    assert ax.get_title() == "title"


def test_plot_caltrack_candidate_qualified():
    candidate_model = DailyModel.from_dict({
        "submodels": {
            'fw-su_sh_wi': {
                "coefficients": {
                    "model_type": "tidd",
                    "intercept": 1,
                },
            "temperature_constraints": {
                "T_min": -100,
                "T_min_seg": -100,     
                "T_max": 200,
                "T_max_seg": 200,     
            },
            "f_unc": 1.0,
            }
        },
        "metrics": None,
        "settings": None
    })
    ax = candidate_model.plot(title="title")
    data = ax.lines[0].get_xydata()
    assert data.shape == (60, 2)
    assert ax.get_title() == "title"


def test_plot_caltrack_candidate_cdd_hdd_model():
    candidate_model = DailyModel.from_2_0_dict({
        "model_type": "cdd_hdd",
        "formula":"formula",
        "status":"QUALIFIED",
        "model_params":{
            "beta_hdd": 1,
            "beta_cdd": 1,
            "cooling_balance_point": 65,
            "heating_balance_point": 65,
            "intercept": 1,
        },
    })
    ax = candidate_model.plot()
    data = ax.lines[0].get_xydata()
    assert data.shape == (60, 2)