#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

   Copyright 2018 Open Energy Efficiency, Inc.

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

from eemeter.caltrack.usage_per_day import (
    CalTRACKUsagePerDayCandidateModel,
    CalTRACKUsagePerDayModelResults,
)
from eemeter.visualization import plot_energy_signature, plot_time_series


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
    assert data.shape == (809, 2)
    assert ax.get_title() == "title"


def test_plot_caltrack_candidate_qualified():
    candidate_model = CalTRACKUsagePerDayCandidateModel(
        model_type="intercept_only",
        formula="formula",
        status="QUALIFIED",
        model_params={"intercept": 1},
    )
    ax = candidate_model.plot(candidate_model, title="title")
    data = ax.lines[0].get_xydata()
    assert data.shape == (60, 2)
    assert ax.get_title() == "title"


def test_plot_caltrack_candidate_disqualified():
    candidate_model = CalTRACKUsagePerDayCandidateModel(
        model_type="intercept_only",
        formula="formula",
        status="DISQUALIFIED",
        model_params={"intercept": 1},
    )
    ax = candidate_model.plot()
    data = ax.lines[0].get_xydata()
    assert data.shape == (60, 2)


def test_plot_caltrack_candidate_with_range():
    candidate_model = CalTRACKUsagePerDayCandidateModel(
        model_type="intercept_only",
        formula="formula",
        status="QUALIFIED",
        model_params={"intercept": 1},
    )
    ax = candidate_model.plot(temp_range=(10, 20))
    data = ax.lines[0].get_xydata()
    assert data.shape == (10, 2)


def test_plot_caltrack_candidate_best():
    candidate_model = CalTRACKUsagePerDayCandidateModel(
        model_type="intercept_only",
        formula="formula",
        status="QUALIFIED",
        model_params={"intercept": 1},
    )
    ax = candidate_model.plot(best=True)
    data = ax.lines[0].get_xydata()
    assert data.shape == (60, 2)


def test_plot_caltrack_candidate_error():
    candidate_model = CalTRACKUsagePerDayCandidateModel(
        model_type="intercept_only",
        formula="formula",
        status="ERROR",
        model_params={"intercept": 1},
    )
    ax = candidate_model.plot()
    assert ax is None


def test_plot_caltrack_candidate_cdd_hdd_model():
    candidate_model = CalTRACKUsagePerDayCandidateModel(
        model_type="cdd_hdd",
        formula="formula",
        status="QUALIFIED",
        model_params={
            "beta_hdd": 1,
            "beta_cdd": 1,
            "cooling_balance_point": 65,
            "heating_balance_point": 65,
            "intercept": 1,
        },
    )
    ax = candidate_model.plot()
    data = ax.lines[0].get_xydata()
    assert data.shape == (60, 2)


def test_plot_model_results(il_electricity_cdd_hdd_daily):
    candidate_model = CalTRACKUsagePerDayCandidateModel(
        model_type="intercept_only",
        formula="formula",
        status="QUALIFIED",
        model_params={"intercept": 1},
    )
    model_results = CalTRACKUsagePerDayModelResults(
        status="status",
        method_name="method_name",
        model=candidate_model,
        candidates=[candidate_model],
    )
    ax = model_results.plot(title="title", with_candidates=True)
    data = ax.lines[0].get_xydata()
    assert data.shape == (70, 2)
    data = ax.lines[1].get_xydata()
    assert data.shape == (70, 2)
    assert ax.get_title() == "title"
