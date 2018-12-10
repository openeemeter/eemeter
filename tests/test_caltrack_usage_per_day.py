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
import json

import numpy as np
import pandas as pd
import pytest

from eemeter.caltrack.usage_per_day import (
    CalTRACKUsagePerDayCandidateModel,
    CalTRACKUsagePerDayModelResults,
    DataSufficiency,
    _caltrack_predict_design_matrix,
    fit_caltrack_usage_per_day_model,
    caltrack_usage_per_day_predict,
    caltrack_sufficiency_criteria,
    get_intercept_only_candidate_models,
    get_too_few_non_zero_degree_day_warning,
    get_total_degree_day_too_low_warning,
    get_parameter_negative_warning,
    get_parameter_p_value_too_high_warning,
    get_cdd_only_candidate_models,
    get_hdd_only_candidate_models,
    get_cdd_hdd_candidate_models,
    select_best_candidate,
)
from eemeter.exceptions import MissingModelParameterError, UnrecognizedModelTypeError
from eemeter.features import (
    compute_time_features,
    compute_temperature_features,
    compute_usage_per_day_feature,
    merge_features,
)
from eemeter.metrics import ModelMetrics
from eemeter.warnings import EEMeterWarning
from eemeter.transform import day_counts, get_baseline_data


def test_candidate_model_minimal():
    candidate_model = CalTRACKUsagePerDayCandidateModel(
        model_type="model_type", formula="formula", status="status"
    )
    assert candidate_model.model_type == "model_type"
    assert candidate_model.formula == "formula"
    assert candidate_model.status == "status"
    assert candidate_model.model_params == {}
    assert candidate_model.warnings == []
    assert str(candidate_model).startswith("CalTRACKUsagePerDayCandidateModel")
    assert candidate_model.json() == {
        "formula": "formula",
        "model_params": {},
        "model_type": "model_type",
        "r_squared_adj": None,
        "status": "status",
        "warnings": [],
    }


def test_candidate_model_json_with_warning():
    eemeter_warning = EEMeterWarning(
        qualified_name="qualified_name", description="description", data={}
    )
    candidate_model = CalTRACKUsagePerDayCandidateModel(
        model_type="model_type",
        formula="formula",
        status="status",
        warnings=[eemeter_warning],
    )
    assert candidate_model.json() == {
        "formula": "formula",
        "model_params": {},
        "model_type": "model_type",
        "r_squared_adj": None,
        "status": "status",
        "warnings": [
            {
                "data": {},
                "description": "description",
                "qualified_name": "qualified_name",
            }
        ],
    }


def test_candidate_model_json_none_and_nan_values():
    eemeter_warning = EEMeterWarning(
        qualified_name="qualified_name", description="description", data={}
    )
    candidate_model = CalTRACKUsagePerDayCandidateModel(
        model_type="model_type",
        formula="formula",
        status="status",
        warnings=[eemeter_warning],
        r_squared_adj=None,
    )
    assert candidate_model.json()["r_squared_adj"] is None
    candidate_model.r_squared_adj = np.nan
    assert candidate_model.json()["r_squared_adj"] is None


def test_data_sufficiency_minimal():
    data_sufficiency = DataSufficiency(status="status", criteria_name="criteria_name")
    assert data_sufficiency.status == "status"
    assert data_sufficiency.criteria_name == "criteria_name"
    assert data_sufficiency.warnings == []
    assert data_sufficiency.settings == {}
    assert str(data_sufficiency).startswith("DataSufficiency")
    assert data_sufficiency.json() == {
        "criteria_name": "criteria_name",
        "settings": {},
        "status": "status",
        "warnings": [],
    }


def test_data_sufficiency_json_with_warning():
    eemeter_warning = EEMeterWarning(
        qualified_name="qualified_name", description="description", data={}
    )
    data_sufficiency = DataSufficiency(
        status="status", criteria_name="criteria_name", warnings=[eemeter_warning]
    )
    assert data_sufficiency.json() == {
        "criteria_name": "criteria_name",
        "settings": {},
        "status": "status",
        "warnings": [
            {
                "data": {},
                "description": "description",
                "qualified_name": "qualified_name",
            }
        ],
    }


def test_model_results_minimal():
    model_results = CalTRACKUsagePerDayModelResults(
        status="status", method_name="method_name"
    )
    assert model_results.status == "status"
    assert model_results.method_name == "method_name"
    assert model_results.model is None
    assert model_results.r_squared_adj is None
    assert model_results.candidates == []
    assert model_results.warnings == []
    assert model_results.metadata == {}
    assert model_results.settings == {}
    assert str(model_results).startswith("CalTRACKUsagePerDayModelResults")
    assert model_results.json() == {
        "candidates": None,
        "interval": None,
        "metadata": {},
        "method_name": "method_name",
        "totals_metrics": None,
        "avgs_metrics": None,
        "model": None,
        "settings": {},
        "status": "status",
        "r_squared_adj": None,
        "warnings": [],
    }


def test_model_results_json_with_objects():
    candidate_model = CalTRACKUsagePerDayCandidateModel(
        model_type="model_type", formula="formula", status="status"
    )
    eemeter_warning = EEMeterWarning(
        qualified_name="qualified_name", description="description", data={}
    )
    model_results = CalTRACKUsagePerDayModelResults(
        status="status",
        method_name="method_name",
        model=candidate_model,
        candidates=[candidate_model],
        warnings=[eemeter_warning],
    )
    assert model_results.json(with_candidates=True) == {
        "candidates": [
            {
                "formula": "formula",
                "model_params": {},
                "model_type": "model_type",
                "r_squared_adj": None,
                "status": "status",
                "warnings": [],
            }
        ],
        "metadata": {},
        "interval": None,
        "method_name": "method_name",
        "totals_metrics": None,
        "avgs_metrics": None,
        "model": {
            "formula": "formula",
            "model_params": {},
            "model_type": "model_type",
            "r_squared_adj": None,
            "status": "status",
            "warnings": [],
        },
        "settings": {},
        "status": "status",
        "r_squared_adj": None,
        "warnings": [
            {
                "data": {},
                "description": "description",
                "qualified_name": "qualified_name",
            }
        ],
    }


def test_model_results_json_with_nan_r_squared_adj():
    candidate_model = CalTRACKUsagePerDayCandidateModel(
        model_type="model_type",
        formula="formula",
        status="status",
        r_squared_adj=np.nan,
    )
    model_results = CalTRACKUsagePerDayModelResults(
        status="status",
        method_name="method_name",
        model=candidate_model,
        r_squared_adj=np.nan,
    )
    assert model_results.json() == {
        "candidates": None,
        "interval": None,
        "metadata": {},
        "method_name": "method_name",
        "totals_metrics": None,
        "avgs_metrics": None,
        "model": {
            "formula": "formula",
            "model_params": {},
            "model_type": "model_type",
            "r_squared_adj": None,
            "status": "status",
            "warnings": [],
        },
        "settings": {},
        "status": "status",
        "r_squared_adj": None,
        "warnings": [],
    }


def test_model_results_json_with_model_metrics():
    candidate_model = CalTRACKUsagePerDayCandidateModel(
        model_type="model_type", formula="formula", status="status", r_squared_adj=0.5
    )
    model_results = CalTRACKUsagePerDayModelResults(
        status="status",
        method_name="method_name",
        model=candidate_model,
        r_squared_adj=np.nan,
    )
    model_metrics = ModelMetrics(
        observed_input=pd.Series([0, 1, 2]), predicted_input=pd.Series([1, 0, 2])
    )
    json_result = model_results.json()
    json.dumps(json_result)  # just make sure it's valid json

    assert "totals_metrics" in json_result
    assert "avgs_metrics" in json_result
    json_result["totals_metrics"] = {}  # overwrite because of floats
    json_result["avgs_metrics"] = {}  # overwrite because of floats

    assert json_result == {
        "candidates": None,
        "interval": None,
        "metadata": {},
        "method_name": "method_name",
        "totals_metrics": {},
        "avgs_metrics": {},
        "model": {
            "formula": "formula",
            "model_params": {},
            "model_type": "model_type",
            "r_squared_adj": 0.5,
            "status": "status",
            "warnings": [],
        },
        "settings": {},
        "status": "status",
        "r_squared_adj": None,
        "warnings": [],
    }


@pytest.fixture
def utc_index():
    return pd.date_range("2011-01-01", freq="H", periods=365 * 24 + 1, tz="UTC")


@pytest.fixture
def temperature_data(utc_index):
    series = pd.Series(
        [
            30.0 * ((i % (365 * 24.0)) / (365 * 24.0))  # 30 * frac of way through year
            + 50.0  # range from 50 to 80
            for i in range(len(utc_index))
        ],
        index=utc_index,
    )
    return series


@pytest.fixture
def prediction_index(temperature_data):
    return temperature_data.resample("D").mean().index


@pytest.fixture
def candidate_model_no_model_none():
    return CalTRACKUsagePerDayCandidateModel(
        model_type=None,
        formula="formula",
        status="QUALIFIED",
        model_params={"intercept": 1},
    )


def test_caltrack_predict_no_model_none(
    candidate_model_no_model_none, prediction_index, temperature_data
):
    with pytest.raises(ValueError):
        candidate_model_no_model_none.predict(prediction_index, temperature_data)


@pytest.fixture
def candidate_model_intercept_only():
    return CalTRACKUsagePerDayCandidateModel(
        model_type="intercept_only",
        formula="formula",
        status="QUALIFIED",
        model_params={"intercept": 1},
    )


def test_caltrack_predict_intercept_only(
    candidate_model_intercept_only, prediction_index, temperature_data
):
    model_prediction = candidate_model_intercept_only.predict(
        prediction_index, temperature_data
    )
    prediction = model_prediction.result
    assert prediction["predicted_usage"].sum() == 365
    assert sorted(prediction.columns) == ["predicted_usage"]


def test_caltrack_predict_intercept_only_with_disaggregated(
    candidate_model_intercept_only, prediction_index, temperature_data
):
    model_prediction = candidate_model_intercept_only.predict(
        prediction_index, temperature_data, with_disaggregated=True
    )
    prediction = model_prediction.result
    assert prediction["base_load"].sum() == 365.0
    assert prediction["cooling_load"].sum() == 0.0
    assert prediction["heating_load"].sum() == 0.0
    assert prediction["predicted_usage"].sum() == 365.0
    assert sorted(prediction.columns) == [
        "base_load",
        "cooling_load",
        "heating_load",
        "predicted_usage",
    ]


def test_caltrack_predict_intercept_only_with_design_matrix(
    candidate_model_intercept_only, prediction_index, temperature_data
):
    model_prediction = candidate_model_intercept_only.predict(
        prediction_index, temperature_data, with_design_matrix=True
    )
    prediction = model_prediction.result
    assert sorted(prediction.columns) == [
        "n_days",
        "n_days_dropped",
        "n_days_kept",
        "predicted_usage",
        "temperature_mean",
    ]
    assert prediction.n_days.sum() == 366.0
    assert prediction.n_days_dropped.sum() == 1
    assert prediction.n_days_kept.sum() == 365
    assert prediction.predicted_usage.sum() == 365.0
    assert round(prediction.temperature_mean.mean()) == 65.0


@pytest.fixture
def candidate_model_missing_params():
    return CalTRACKUsagePerDayCandidateModel(
        model_type="intercept_only",
        formula="formula",
        status="QUALIFIED",
        model_params={},
    )


def test_caltrack_predict_missing_params(
    candidate_model_missing_params, prediction_index, temperature_data
):
    with pytest.raises(MissingModelParameterError):
        candidate_model_missing_params.predict(prediction_index, temperature_data)


@pytest.fixture
def candidate_model_cdd_only():
    return CalTRACKUsagePerDayCandidateModel(
        model_type="cdd_only",
        formula="formula",
        status="QUALIFIED",
        model_params={"intercept": 1, "beta_cdd": 1, "cooling_balance_point": 65},
    )


def test_caltrack_predict_cdd_only(
    candidate_model_cdd_only, prediction_index, temperature_data
):
    model_prediction = candidate_model_cdd_only.predict(
        prediction_index, temperature_data
    )
    prediction_df = model_prediction.result
    assert round(prediction_df.predicted_usage.sum()) == 1733


def test_caltrack_predict_cdd_only_with_disaggregated(
    candidate_model_cdd_only, prediction_index, temperature_data
):
    model_prediction = candidate_model_cdd_only.predict(
        prediction_index, temperature_data, with_disaggregated=True
    )
    prediction = model_prediction.result
    assert round(prediction.predicted_usage.sum()) == 1733
    assert round(prediction.base_load.sum()) == 365.0
    assert round(prediction.heating_load.sum()) == 0.0
    assert round(prediction.cooling_load.sum()) == 1368.0


def test_caltrack_predict_cdd_only_with_design_matrix(
    candidate_model_cdd_only, prediction_index, temperature_data
):
    model_prediction = candidate_model_cdd_only.predict(
        prediction_index, temperature_data, with_design_matrix=True
    )
    prediction = model_prediction.result
    assert sorted(prediction.columns) == [
        "cdd_65",
        "n_days",
        "n_days_dropped",
        "n_days_kept",
        "predicted_usage",
        "temperature_mean",
    ]
    assert round(prediction.cdd_65.sum()) == 1368.0
    assert prediction.n_days.sum() == 365.0
    assert prediction.n_days_dropped.sum() == 0
    assert prediction.n_days_kept.sum() == 365.0
    assert round(prediction.predicted_usage.sum()) == 1733
    assert round(prediction.temperature_mean.mean()) == 65.0


@pytest.fixture
def candidate_model_hdd_only():
    return CalTRACKUsagePerDayCandidateModel(
        model_type="hdd_only",
        formula="formula",
        status="QUALIFIED",
        model_params={"intercept": 1, "beta_hdd": 1, "heating_balance_point": 65},
    )


def test_caltrack_predict_hdd_only(
    candidate_model_hdd_only, prediction_index, temperature_data
):
    model_prediction = candidate_model_hdd_only.predict(
        prediction_index, temperature_data
    )
    prediction = model_prediction.result
    assert round(prediction.predicted_usage.sum()) == 1734


def test_caltrack_predict_hdd_only_with_disaggregated(
    candidate_model_hdd_only, prediction_index, temperature_data
):
    model_prediction = candidate_model_hdd_only.predict(
        prediction_index, temperature_data, with_disaggregated=True
    )
    prediction = model_prediction.result
    assert round(prediction.predicted_usage.sum()) == 1734
    assert round(prediction.base_load.sum()) == 365.0
    assert round(prediction.heating_load.sum()) == 1369.0
    assert round(prediction.cooling_load.sum()) == 0.0


def test_caltrack_predict_hdd_only_with_design_matrix(
    candidate_model_hdd_only, prediction_index, temperature_data
):
    model_prediction = candidate_model_hdd_only.predict(
        prediction_index, temperature_data, with_design_matrix=True
    )
    prediction = model_prediction.result
    assert sorted(prediction.columns) == [
        "hdd_65",
        "n_days",
        "n_days_dropped",
        "n_days_kept",
        "predicted_usage",
        "temperature_mean",
    ]
    assert round(prediction.hdd_65.sum()) == 1369.0
    assert prediction.n_days.sum() == 365.0
    assert prediction.n_days_dropped.sum() == 0
    assert prediction.n_days_kept.sum() == 365.0
    assert round(prediction.predicted_usage.sum()) == 1734
    assert round(prediction.temperature_mean.mean()) == 65.0


@pytest.fixture
def candidate_model_cdd_hdd():
    return CalTRACKUsagePerDayCandidateModel(
        model_type="cdd_hdd",
        formula="formula",
        status="QUALIFIED",
        model_params={
            "intercept": 1,
            "beta_hdd": 1,
            "heating_balance_point": 60,
            "beta_cdd": 1,
            "cooling_balance_point": 70,
        },
    )


def test_caltrack_predict_cdd_hdd(
    candidate_model_cdd_hdd, prediction_index, temperature_data
):
    model_prediction = candidate_model_cdd_hdd.predict(
        prediction_index, temperature_data
    )
    prediction = model_prediction.result
    assert round(prediction.predicted_usage.sum()) == 1582.0


def test_caltrack_predict_cdd_hdd_disaggregated(
    candidate_model_cdd_hdd, prediction_index, temperature_data
):
    model_prediction = candidate_model_cdd_hdd.predict(
        prediction_index, temperature_data, with_disaggregated=True
    )
    prediction = model_prediction.result
    assert round(prediction.predicted_usage.sum()) == 1582.0
    assert round(prediction.base_load.sum()) == 365.0
    assert round(prediction.heating_load.sum()) == 609.0
    assert round(prediction.cooling_load.sum()) == 608.0


def test_caltrack_predict_cdd_hdd_with_design_matrix(
    candidate_model_cdd_hdd, prediction_index, temperature_data
):
    model_prediction = candidate_model_cdd_hdd.predict(
        prediction_index, temperature_data, with_design_matrix=True
    )
    prediction = model_prediction.result
    assert sorted(prediction.columns) == [
        "cdd_70",
        "hdd_60",
        "n_days",
        "n_days_dropped",
        "n_days_kept",
        "predicted_usage",
        "temperature_mean",
    ]
    assert round(prediction.cdd_70.sum()) == 608.0
    assert round(prediction.hdd_60.sum()) == 609.0
    assert prediction.n_days.sum() == 365.0
    assert prediction.n_days_dropped.sum() == 0
    assert prediction.n_days_kept.sum() == 365.0
    assert round(prediction.predicted_usage.sum()) == 1582.0
    assert round(prediction.temperature_mean.mean()) == 65.0


@pytest.fixture
def candidate_model_bad_model_type():
    return CalTRACKUsagePerDayCandidateModel(
        model_type="unknown", formula="formula", status="QUALIFIED", model_params={}
    )


def test_caltrack_predict_bad_model_type(
    candidate_model_bad_model_type, temperature_data, prediction_index
):
    with pytest.raises(UnrecognizedModelTypeError):
        candidate_model_bad_model_type.predict(prediction_index, temperature_data)


def test_caltrack_predict_empty(
    candidate_model_bad_model_type, temperature_data, prediction_index
):
    model_prediction_obj = candidate_model_bad_model_type.predict(
        prediction_index[:0], temperature_data[:0]
    )
    assert model_prediction_obj.result.empty is True


@pytest.fixture
def cdd_hdd_h54_c67_billing_monthly_totals(il_electricity_cdd_hdd_billing_monthly):
    meter_data = il_electricity_cdd_hdd_billing_monthly["meter_data"]
    temperature_data = il_electricity_cdd_hdd_billing_monthly["temperature_data"]
    temperature_features = compute_temperature_features(
        meter_data.index,
        temperature_data,
        heating_balance_points=[54],
        cooling_balance_points=[67],
        use_mean_daily_values=False,
    )
    data = merge_features([meter_data, temperature_features])
    return data


def test_caltrack_predict_design_matrix_input_avg_false_output_avg_true(
    cdd_hdd_h54_c67_billing_monthly_totals
):
    data = cdd_hdd_h54_c67_billing_monthly_totals
    prediction = _caltrack_predict_design_matrix(
        "cdd_hdd",
        {
            "intercept": 13.420093629452852,
            "beta_cdd": 2.257868665412409,
            "beta_hdd": 1.0479347638717025,
            "cooling_balance_point": 67,
            "heating_balance_point": 54,
        },
        data,
        input_averages=False,
        output_averages=True,
    )
    assert round(prediction.mean(), 3) == 28.253


def test_caltrack_predict_design_matrix_input_avg_false_output_avg_false(
    cdd_hdd_h54_c67_billing_monthly_totals
):
    data = cdd_hdd_h54_c67_billing_monthly_totals
    prediction = _caltrack_predict_design_matrix(
        "cdd_hdd",
        {
            "intercept": 13.420093629452852,
            "beta_cdd": 2.257868665412409,
            "beta_hdd": 1.0479347638717025,
            "cooling_balance_point": 67,
            "heating_balance_point": 54,
        },
        data,
        input_averages=False,
        output_averages=False,
    )
    assert round(prediction.mean(), 3) == 855.832


@pytest.fixture
def cdd_hdd_h54_c67_billing_monthly_avgs(il_electricity_cdd_hdd_billing_monthly):
    meter_data = il_electricity_cdd_hdd_billing_monthly["meter_data"]
    temperature_data = il_electricity_cdd_hdd_billing_monthly["temperature_data"]
    temperature_features = compute_temperature_features(
        meter_data.index,
        temperature_data,
        heating_balance_points=[54],
        cooling_balance_points=[67],
        use_mean_daily_values=True,
    )
    meter_data_feature = compute_usage_per_day_feature(meter_data)
    data = merge_features([meter_data_feature, temperature_features])
    return data


def test_caltrack_predict_design_matrix_input_avg_true_output_avg_false(
    cdd_hdd_h54_c67_billing_monthly_avgs
):
    data = cdd_hdd_h54_c67_billing_monthly_avgs
    prediction = _caltrack_predict_design_matrix(
        "cdd_hdd",
        {
            "intercept": 13.420093629452852,
            "beta_cdd": 2.257868665412409,
            "beta_hdd": 1.0479347638717025,
            "cooling_balance_point": 67,
            "heating_balance_point": 54,
        },
        data,
        input_averages=True,
        output_averages=False,
    )
    assert round(prediction.mean(), 3) == 855.832


def test_caltrack_predict_design_matrix_input_avg_true_output_avg_true(
    cdd_hdd_h54_c67_billing_monthly_avgs
):
    data = cdd_hdd_h54_c67_billing_monthly_avgs
    prediction = _caltrack_predict_design_matrix(
        "cdd_hdd",
        {
            "intercept": 13.420093629452852,
            "beta_cdd": 2.257868665412409,
            "beta_hdd": 1.0479347638717025,
            "cooling_balance_point": 67,
            "heating_balance_point": 54,
        },
        data,
        input_averages=True,
        output_averages=True,
    )
    assert round(prediction.mean(), 3) == 28.253


def test_caltrack_predict_design_matrix_n_days(cdd_hdd_h54_c67_billing_monthly_totals):
    # This makes sure that the method works with n_days when
    # DatetimeIndexes are not available.
    data = cdd_hdd_h54_c67_billing_monthly_totals
    data = data.reset_index(drop=True)
    data["n_days"] = 1
    prediction = _caltrack_predict_design_matrix(
        "cdd_hdd",
        {
            "intercept": 13.420093629452852,
            "beta_cdd": 2.257868665412409,
            "beta_hdd": 1.0479347638717025,
            "cooling_balance_point": 67,
            "heating_balance_point": 54,
        },
        data,
        input_averages=True,
        output_averages=True,
    )
    assert prediction.mean() is not None


def test_caltrack_predict_design_matrix_no_days_fails(
    cdd_hdd_h54_c67_billing_monthly_totals
):
    # This makes sure that the method fails if neither n_days nor
    # a DatetimeIndex is available.
    data = cdd_hdd_h54_c67_billing_monthly_totals
    data = data.reset_index(drop=True)
    with pytest.raises(ValueError):
        _caltrack_predict_design_matrix(
            "cdd_hdd",
            {
                "intercept": 13.420093629452852,
                "beta_cdd": 2.257868665412409,
                "beta_hdd": 1.0479347638717025,
                "cooling_balance_point": 67,
                "heating_balance_point": 54,
            },
            data,
            input_averages=True,
            output_averages=True,
        )


def test_get_too_few_non_zero_degree_day_warning_ok():
    warnings = get_too_few_non_zero_degree_day_warning(
        model_type="model_type",
        balance_point=65,
        degree_day_type="xdd",
        degree_days=pd.Series([1, 1, 1]),
        minimum_non_zero=2,
    )
    assert warnings == []


def test_get_too_few_non_zero_degree_day_warning_fail():
    warnings = get_too_few_non_zero_degree_day_warning(
        model_type="model_type",
        balance_point=65,
        degree_day_type="xdd",
        degree_days=pd.Series([0, 0, 3]),
        minimum_non_zero=2,
    )
    assert len(warnings) == 1
    warning = warnings[0]
    assert warning.qualified_name == (
        "eemeter.caltrack_daily.model_type.too_few_non_zero_xdd"
    )
    assert warning.description == (
        "Number of non-zero daily XDD values below accepted minimum."
        " Candidate fit not attempted."
    )
    assert warning.data == {
        "minimum_non_zero_xdd": 2,
        "n_non_zero_xdd": 1,
        "xdd_balance_point": 65,
    }


def test_get_total_degree_day_too_low_warning_ok():
    warnings = get_total_degree_day_too_low_warning(
        model_type="model_type",
        balance_point=65,
        degree_day_type="xdd",
        avg_degree_days=pd.Series([1, 1, 1]),
        period_days=pd.Series([3, 1, 2]),
        minimum_total=4,
    )
    assert warnings == []


def test_get_total_degree_day_too_low_warning_fail():
    warnings = get_total_degree_day_too_low_warning(
        model_type="model_type",
        balance_point=65,
        degree_day_type="xdd",
        avg_degree_days=pd.Series([0.5, 0.5, 0.5]),
        period_days=pd.Series([3, 1, 2]),
        minimum_total=4,
    )
    assert len(warnings) == 1
    warning = warnings[0]
    assert warning.qualified_name == (
        "eemeter.caltrack_daily.model_type.total_xdd_too_low"
    )
    assert warning.description == (
        "Total XDD below accepted minimum. Candidate fit not attempted."
    )
    assert warning.data == {
        "total_xdd": 3.0,
        "total_xdd_minimum": 4,
        "xdd_balance_point": 65,
    }


def test_get_parameter_negative_warning_ok():
    warnings = get_parameter_negative_warning(
        "intercept_only", {"intercept": 0}, "intercept"
    )
    assert warnings == []


def test_get_parameter_negative_warning_fail():
    warnings = get_parameter_negative_warning(
        "intercept_only", {"intercept": -1}, "intercept"
    )
    assert len(warnings) == 1
    warning = warnings[0]
    assert warning.qualified_name == (
        "eemeter.caltrack_daily.intercept_only.intercept_negative"
    )
    assert warning.description == (
        "Model fit intercept parameter is negative. Candidate model rejected."
    )
    assert warning.data == {"intercept": -1}


def test_get_parameter_p_value_too_high_warning_ok():
    warnings = get_parameter_p_value_too_high_warning(
        "intercept_only", {"intercept": 0}, "intercept", 0.1, 0.1
    )
    assert warnings == []


def test_get_parameter_p_value_too_high_warning_fail():
    warnings = get_parameter_p_value_too_high_warning(
        "intercept_only", {"intercept": 0}, "intercept", 0.2, 0.1
    )
    assert len(warnings) == 1
    warning = warnings[0]
    assert warning.qualified_name == (
        "eemeter.caltrack_daily.intercept_only.intercept_p_value_too_high"
    )
    assert warning.description == (
        "Model fit intercept p-value is too high. Candidate model rejected."
    )
    assert warning.data == {
        "intercept": 0,
        "intercept_maximum_p_value": 0.1,
        "intercept_p_value": 0.2,
    }


def test_get_intercept_only_candidate_models_fail():
    # should be covered by ETL, but this ensures no negative values.
    data = pd.DataFrame({"meter_value": np.arange(10) * -1})
    candidate_models = get_intercept_only_candidate_models(data, weights_col=None)
    assert len(candidate_models) == 1
    model = candidate_models[0]
    assert model.model_type == "intercept_only"
    assert model.formula == "meter_value ~ 1"
    assert model.status == "DISQUALIFIED"
    assert model.model is not None
    assert model.result is not None
    assert list(sorted(model.model_params.keys())) == ["intercept"]
    assert round(model.model_params["intercept"], 2) == -4.5
    assert model.r_squared_adj == 0
    assert len(model.warnings) == 1
    warning = model.warnings[0]
    assert warning.qualified_name == (
        "eemeter.caltrack_daily.intercept_only.intercept_negative"
    )


def test_get_intercept_only_candidate_models_qualified(
    prediction_index, temperature_data
):
    data = pd.DataFrame({"meter_value": np.arange(10)})
    candidate_models = get_intercept_only_candidate_models(data, weights_col=None)
    assert len(candidate_models) == 1
    model = candidate_models[0]
    assert model.model_type == "intercept_only"
    assert model.formula == "meter_value ~ 1"
    assert model.status == "QUALIFIED"
    assert model.model is not None
    assert model.result is not None
    assert list(sorted(model.model_params.keys())) == ["intercept"]
    assert round(model.model_params["intercept"], 2) == 4.5
    model_prediction = model.predict(prediction_index, temperature_data)
    prediction = model_prediction.result
    assert round(prediction.predicted_usage.sum(), 2) == 1642.5
    assert model.r_squared_adj == 0
    assert model.warnings == []
    assert json.dumps(model.json()) is not None


def test_get_intercept_only_candidate_models_qualified_with_weights(
    prediction_index, temperature_data
):
    data = pd.DataFrame({"meter_value": np.arange(10), "weights": np.arange(10)})
    candidate_models = get_intercept_only_candidate_models(data, "weights")
    assert len(candidate_models) == 1
    model = candidate_models[0]
    assert model.model_type == "intercept_only"
    assert model.formula == "meter_value ~ 1"
    assert model.status == "QUALIFIED"
    assert model.model is not None
    assert model.result is not None
    assert list(sorted(model.model_params.keys())) == ["intercept"]
    assert round(model.model_params["intercept"], 2) == 6.33
    model_prediction = model.predict(prediction_index, temperature_data)
    prediction = model_prediction.result
    assert round(prediction.predicted_usage.sum(), 2) == 2311.67
    assert model.r_squared_adj == 0
    assert model.warnings == []
    assert json.dumps(model.json()) is not None


def test_get_intercept_only_candidate_models_error():
    data = pd.DataFrame({"meter_value": []})
    candidate_models = get_intercept_only_candidate_models(data, weights_col=None)
    assert len(candidate_models) == 1
    model = candidate_models[0]
    assert len(model.warnings) == 1
    warning = model.warnings[0]
    assert warning.qualified_name == (
        "eemeter.caltrack_daily.intercept_only.model_results"
    )
    assert warning.description == (
        "Error encountered in statsmodels.formula.api.ols method." " (Empty data?)"
    )
    assert list(sorted(warning.data.keys())) == ["traceback"]
    assert warning.data["traceback"] is not None


def test_get_cdd_only_candidate_models_qualified(prediction_index, temperature_data):
    data = pd.DataFrame({"meter_value": [1, 1, 1, 6], "cdd_65": [0, 0.1, 0, 5]})
    candidate_models = get_cdd_only_candidate_models(data, 1, 1, 0.1, None)
    assert len(candidate_models) == 1
    model = candidate_models[0]
    assert model.model_type == "cdd_only"
    assert model.formula == "meter_value ~ cdd_65"
    assert model.status == "QUALIFIED"
    assert model.model is not None
    assert model.result is not None
    assert list(sorted(model.model_params.keys())) == [
        "beta_cdd",
        "cooling_balance_point",
        "intercept",
    ]
    assert round(model.model_params["beta_cdd"], 2) == 1.01
    assert round(model.model_params["cooling_balance_point"], 2) == 65
    assert round(model.model_params["intercept"], 2) == 0.97
    model_prediction = model.predict(prediction_index, temperature_data)
    prediction = model_prediction.result
    assert round(prediction.predicted_usage.sum(), 2) == 1730.04
    assert round(model.r_squared_adj, 2) == 1.00
    assert model.warnings == []
    assert json.dumps(model.json()) is not None


def test_get_cdd_only_candidate_models_qualified_with_weights(
    prediction_index, temperature_data
):
    data = pd.DataFrame(
        {
            "meter_value": [1, 1, 1, 6],
            "cdd_65": [0, 0.1, 0, 5],
            "weights": [1, 100, 1, 1],
        }
    )
    candidate_models = get_cdd_only_candidate_models(data, 1, 1, 0.1, "weights")
    assert len(candidate_models) == 1
    model = candidate_models[0]
    assert model.model_type == "cdd_only"
    assert model.formula == "meter_value ~ cdd_65"
    assert model.status == "QUALIFIED"
    assert model.model is not None
    assert model.result is not None
    assert list(sorted(model.model_params.keys())) == [
        "beta_cdd",
        "cooling_balance_point",
        "intercept",
    ]
    assert round(model.model_params["beta_cdd"], 2) == 1.02
    assert round(model.model_params["cooling_balance_point"], 2) == 65
    assert round(model.model_params["intercept"], 2) == 0.9
    model_prediction = model.predict(prediction_index, temperature_data)
    prediction = model_prediction.result
    assert round(prediction.predicted_usage.sum(), 2) == 1723.19
    assert round(model.r_squared_adj, 2) == 1.00
    assert model.warnings == []
    assert json.dumps(model.json()) is not None


def test_get_cdd_only_candidate_models_not_attempted():
    data = pd.DataFrame({"meter_value": [1, 1, 1, 6], "cdd_65": [0, 0.1, 0, 5]})
    candidate_models = get_cdd_only_candidate_models(data, 10, 10, 0.1, None)
    assert len(candidate_models) == 1
    model = candidate_models[0]
    assert model.model_type == "cdd_only"
    assert model.formula == "meter_value ~ cdd_65"
    assert model.status == "NOT ATTEMPTED"
    assert model.model is None
    assert model.result is None
    assert model.model_params == {}
    assert model.r_squared_adj is None
    assert len(model.warnings) == 2
    assert json.dumps(model.json()) is not None


def test_get_cdd_only_candidate_models_disqualified(prediction_index, temperature_data):
    data = pd.DataFrame({"meter_value": [1, 1, 1, -4], "cdd_65": [0, 0.1, 0, 5]})
    candidate_models = get_cdd_only_candidate_models(data, 1, 1, 0.0, None)
    assert len(candidate_models) == 1
    model = candidate_models[0]
    assert model.model_type == "cdd_only"
    assert model.formula == "meter_value ~ cdd_65"
    assert model.status == "DISQUALIFIED"
    assert model.model is not None
    assert model.result is not None
    assert list(sorted(model.model_params.keys())) == [
        "beta_cdd",
        "cooling_balance_point",
        "intercept",
    ]
    assert round(model.model_params["beta_cdd"], 2) == -1.01
    assert round(model.model_params["cooling_balance_point"], 2) == 65
    assert round(model.model_params["intercept"], 2) == 1.03
    model_prediction = model.predict(prediction_index, temperature_data)
    prediction = model_prediction.result
    assert round(prediction.predicted_usage.sum(), 2) == -1000.04
    assert round(model.r_squared_adj, 2) == 1.00
    assert len(model.warnings) == 2
    assert json.dumps(model.json()) is not None


def test_get_cdd_only_candidate_models_error():
    data = pd.DataFrame({"meter_value": [], "cdd_65": []})
    candidate_models = get_cdd_only_candidate_models(data, 0, 0, 0.1, None)
    assert len(candidate_models) == 1
    model = candidate_models[0]
    assert len(model.warnings) == 1
    warning = model.warnings[0]
    assert warning.qualified_name == ("eemeter.caltrack_daily.cdd_only.model_results")
    assert warning.description == (
        "Error encountered in statsmodels.formula.api.ols method." " (Empty data?)"
    )
    assert list(sorted(warning.data.keys())) == ["traceback"]
    assert warning.data["traceback"] is not None


def test_get_hdd_only_candidate_models_qualified(prediction_index, temperature_data):
    data = pd.DataFrame({"meter_value": [1, 1, 1, 6], "hdd_65": [0, 0.1, 0, 5]})
    candidate_models = get_hdd_only_candidate_models(data, 1, 1, 0.1, None)
    assert len(candidate_models) == 1
    model = candidate_models[0]
    assert model.model_type == "hdd_only"
    assert model.formula == "meter_value ~ hdd_65"
    assert model.status == "QUALIFIED"
    assert model.model is not None
    assert model.result is not None
    assert list(sorted(model.model_params.keys())) == [
        "beta_hdd",
        "heating_balance_point",
        "intercept",
    ]
    assert round(model.model_params["beta_hdd"], 2) == 1.01
    assert round(model.model_params["heating_balance_point"], 2) == 65
    assert round(model.model_params["intercept"], 2) == 0.97
    model_prediction = model.predict(prediction_index, temperature_data)
    prediction = model_prediction.result
    assert round(prediction.predicted_usage.sum(), 2) == 1730.67
    assert round(model.r_squared_adj, 2) == 1.00
    assert model.warnings == []
    assert json.dumps(model.json()) is not None


def test_get_hdd_only_candidate_models_qualified_with_weights(
    prediction_index, temperature_data
):
    data = pd.DataFrame(
        {
            "meter_value": [1, 1, 1, 6],
            "hdd_65": [0, 0.1, 0, 5],
            "weights": [1, 100, 1, 1],
        }
    )
    candidate_models = get_hdd_only_candidate_models(data, 1, 1, 0.1, "weights")
    assert len(candidate_models) == 1
    model = candidate_models[0]
    assert model.model_type == "hdd_only"
    assert model.formula == "meter_value ~ hdd_65"
    assert model.status == "QUALIFIED"
    assert model.model is not None
    assert model.result is not None
    assert list(sorted(model.model_params.keys())) == [
        "beta_hdd",
        "heating_balance_point",
        "intercept",
    ]
    assert round(model.model_params["beta_hdd"], 2) == 1.02
    assert round(model.model_params["heating_balance_point"], 2) == 65
    assert round(model.model_params["intercept"], 2) == 0.9
    model_prediction = model.predict(prediction_index, temperature_data)
    prediction = model_prediction.result
    assert round(prediction.predicted_usage.sum(), 2) == 1723.83
    assert round(model.r_squared_adj, 2) == 1.00
    assert model.warnings == []
    assert json.dumps(model.json()) is not None


def test_get_hdd_only_candidate_models_not_attempted():
    data = pd.DataFrame({"meter_value": [1, 1, 1, 6], "hdd_65": [0, 0.1, 0, 5]})
    candidate_models = get_hdd_only_candidate_models(data, 10, 10, 0.1, None)
    assert len(candidate_models) == 1
    model = candidate_models[0]
    assert model.model_type == "hdd_only"
    assert model.formula == "meter_value ~ hdd_65"
    assert model.status == "NOT ATTEMPTED"
    assert model.model is None
    assert model.result is None
    assert model.model_params == {}
    assert model.r_squared_adj is None
    assert len(model.warnings) == 2
    assert json.dumps(model.json()) is not None


def test_get_hdd_only_candidate_models_disqualified(prediction_index, temperature_data):
    data = pd.DataFrame({"meter_value": [1, 1, 1, -4], "hdd_65": [0, 0.1, 0, 5]})
    candidate_models = get_hdd_only_candidate_models(data, 1, 1, 0.0, None)
    assert len(candidate_models) == 1
    model = candidate_models[0]
    assert model.model_type == "hdd_only"
    assert model.formula == "meter_value ~ hdd_65"
    assert model.status == "DISQUALIFIED"
    assert model.model is not None
    assert model.result is not None
    assert list(sorted(model.model_params.keys())) == [
        "beta_hdd",
        "heating_balance_point",
        "intercept",
    ]
    assert round(model.model_params["beta_hdd"], 2) == -1.01
    assert round(model.model_params["heating_balance_point"], 2) == 65
    assert round(model.model_params["intercept"], 2) == 1.03
    model_prediction = model.predict(prediction_index, temperature_data)
    prediction = model_prediction.result
    assert round(prediction.predicted_usage.sum(), 2) == -1000.67
    assert round(model.r_squared_adj, 2) == 1.00
    assert len(model.warnings) == 2
    assert json.dumps(model.json()) is not None


def test_get_hdd_only_candidate_models_error():
    data = pd.DataFrame({"meter_value": [], "hdd_65": []})
    candidate_models = get_hdd_only_candidate_models(data, 0, 0, 0.1, None)
    assert len(candidate_models) == 1
    model = candidate_models[0]
    assert len(model.warnings) == 1
    warning = model.warnings[0]
    assert warning.qualified_name == ("eemeter.caltrack_daily.hdd_only.model_results")
    assert warning.description == (
        "Error encountered in statsmodels.formula.api.ols method." " (Empty data?)"
    )
    assert list(sorted(warning.data.keys())) == ["traceback"]
    assert warning.data["traceback"] is not None


def test_get_cdd_hdd_candidate_models_qualified(prediction_index, temperature_data):
    data = pd.DataFrame(
        {
            "meter_value": [6, 1, 1, 6],
            "cdd_65": [5, 0, 0.1, 0],
            "hdd_65": [0, 0.1, 0.1, 5],
        }
    )
    candidate_models = get_cdd_hdd_candidate_models(data, 1, 1, 1, 1, 0.1, 0.1, None)
    assert len(candidate_models) == 1
    model = candidate_models[0]
    assert model.model_type == "cdd_hdd"
    assert model.formula == "meter_value ~ cdd_65 + hdd_65"
    assert model.status == "QUALIFIED"
    assert model.model is not None
    assert model.result is not None
    assert list(sorted(model.model_params.keys())) == [
        "beta_cdd",
        "beta_hdd",
        "cooling_balance_point",
        "heating_balance_point",
        "intercept",
    ]
    assert round(model.model_params["beta_cdd"], 2) == 1.03
    assert round(model.model_params["beta_hdd"], 2) == 1.03
    assert round(model.model_params["cooling_balance_point"], 2) == 65
    assert round(model.model_params["heating_balance_point"], 2) == 65
    assert round(model.model_params["intercept"], 2) == 0.85
    model_prediction = model.predict(prediction_index, temperature_data)
    prediction = model_prediction.result
    assert round(prediction.predicted_usage.sum(), 2) == 3130.31
    assert round(model.r_squared_adj, 2) == 1.00
    assert model.warnings == []
    assert json.dumps(model.json()) is not None


def test_get_cdd_hdd_candidate_models_qualified_with_weights(
    prediction_index, temperature_data
):
    data = pd.DataFrame(
        {
            "meter_value": [6, 1, 1, 6],
            "cdd_65": [5, 0, 0.1, 0],
            "hdd_65": [0, 0.1, 0.1, 5],
            "weights": [1, 1, 100, 1],
        }
    )
    candidate_models = get_cdd_hdd_candidate_models(
        data, 1, 1, 1, 1, 0.1, 0.1, "weights"
    )
    assert len(candidate_models) == 1
    model = candidate_models[0]
    assert model.model_type == "cdd_hdd"
    assert model.formula == "meter_value ~ cdd_65 + hdd_65"
    assert model.status == "QUALIFIED"
    assert model.model is not None
    assert model.result is not None
    assert list(sorted(model.model_params.keys())) == [
        "beta_cdd",
        "beta_hdd",
        "cooling_balance_point",
        "heating_balance_point",
        "intercept",
    ]
    assert round(model.model_params["beta_cdd"], 2) == 1.04
    assert round(model.model_params["beta_hdd"], 2) == 1.04
    assert round(model.model_params["cooling_balance_point"], 2) == 65
    assert round(model.model_params["heating_balance_point"], 2) == 65
    assert round(model.model_params["intercept"], 2) == 0.79
    model_prediction = model.predict(prediction_index, temperature_data)
    prediction = model_prediction.result
    assert round(prediction.predicted_usage.sum(), 2) == 3139.71
    assert round(model.r_squared_adj, 2) == 1.00
    assert model.warnings == []
    assert json.dumps(model.json()) is not None


def test_get_cdd_hdd_candidate_models_not_attempted():
    data = pd.DataFrame(
        {
            "meter_value": [6, 1, 1, 6],
            "cdd_65": [5, 0, 0.1, 0],
            "hdd_65": [0, 0.1, 0, 5],
        }
    )
    candidate_models = get_cdd_hdd_candidate_models(
        data, 10, 10, 10, 10, 0.1, 0.1, None
    )
    assert len(candidate_models) == 1
    model = candidate_models[0]
    assert model.model_type == "cdd_hdd"
    assert model.formula == "meter_value ~ cdd_65 + hdd_65"
    assert model.status == "NOT ATTEMPTED"
    assert model.model is None
    assert model.result is None
    assert model.model_params == {}
    assert model.r_squared_adj is None
    assert len(model.warnings) == 4
    assert json.dumps(model.json()) is not None


def test_get_cdd_hdd_candidate_models_disqualified(prediction_index, temperature_data):
    data = pd.DataFrame(
        {
            "meter_value": [-4, 1, 1, -4],
            "cdd_65": [5, 0, 0.1, 0],
            "hdd_65": [0, 0.1, 0, 5],
        }
    )
    candidate_models = get_cdd_hdd_candidate_models(data, 1, 1, 1, 1, 0.0, 0.0, None)
    assert len(candidate_models) == 1
    model = candidate_models[0]
    assert model.model_type == "cdd_hdd"
    assert model.formula == "meter_value ~ cdd_65 + hdd_65"
    assert model.status == "DISQUALIFIED"
    assert model.model is not None
    assert model.result is not None
    assert list(sorted(model.model_params.keys())) == [
        "beta_cdd",
        "beta_hdd",
        "cooling_balance_point",
        "heating_balance_point",
        "intercept",
    ]
    assert round(model.model_params["beta_cdd"], 2) == -1.02
    assert round(model.model_params["beta_hdd"], 2) == -1.02
    assert round(model.model_params["cooling_balance_point"], 2) == 65
    assert round(model.model_params["heating_balance_point"], 2) == 65
    assert round(model.model_params["intercept"], 2) == 1.1
    model_prediction = model.predict(prediction_index, temperature_data)
    prediction = model_prediction.result
    assert round(prediction.predicted_usage.sum(), 2) == -2391.1
    assert round(model.r_squared_adj, 2) == 1.00
    assert len(model.warnings) == 4
    assert json.dumps(model.json()) is not None


def test_get_cdd_hdd_candidate_models_error():
    data = pd.DataFrame({"meter_value": [], "hdd_65": [], "cdd_65": []})
    candidate_models = get_cdd_hdd_candidate_models(data, 0, 0, 0, 0, 0.1, 0.1, None)
    assert len(candidate_models) == 1
    model = candidate_models[0]
    assert len(model.warnings) == 1
    warning = model.warnings[0]
    assert warning.qualified_name == ("eemeter.caltrack_daily.cdd_hdd.model_results")
    assert warning.description == (
        "Error encountered in statsmodels.formula.api.ols method." " (Empty data?)"
    )
    assert list(sorted(warning.data.keys())) == ["traceback"]
    assert warning.data["traceback"] is not None


@pytest.fixture
def candidate_model_qualified_high_r2():
    return CalTRACKUsagePerDayCandidateModel(
        model_type="model_type", formula="formula1", status="QUALIFIED", r_squared_adj=1
    )


@pytest.fixture
def candidate_model_qualified_low_r2():
    return CalTRACKUsagePerDayCandidateModel(
        model_type="model_type", formula="formula2", status="QUALIFIED", r_squared_adj=0
    )


@pytest.fixture
def candidate_model_disqualified():
    return CalTRACKUsagePerDayCandidateModel(
        model_type="model_type",
        formula="formula3",
        status="DISQUALIFIED",
        r_squared_adj=0.5,
    )


def test_select_best_candidate_ok(
    candidate_model_qualified_high_r2,
    candidate_model_qualified_low_r2,
    candidate_model_disqualified,
):
    candidates = [
        candidate_model_qualified_high_r2,
        candidate_model_qualified_low_r2,
        candidate_model_disqualified,
    ]

    best_candidate, warnings = select_best_candidate(candidates)
    assert warnings == []
    assert best_candidate.status == "QUALIFIED"
    assert best_candidate.formula == "formula1"
    assert best_candidate.r_squared_adj == 1


def test_select_best_candidate_none(candidate_model_disqualified,):
    candidates = [candidate_model_disqualified]

    best_candidate, warnings = select_best_candidate(candidates)
    assert best_candidate is None
    assert len(warnings) == 1
    warning = warnings[0]
    assert warning.qualified_name == (
        "eemeter.caltrack_daily.select_best_candidate.no_candidates"
    )
    assert warning.description == ("No qualified model candidates available.")
    assert warning.data == {"status_count:DISQUALIFIED": 1}


def test_fit_caltrack_usage_per_day_model_empty():
    data = pd.DataFrame({"meter_value": [], "hdd_65": [], "cdd_65": []})
    model_results = fit_caltrack_usage_per_day_model(data)
    assert model_results.method_name == "caltrack_usage_per_day"
    assert model_results.status == "NO DATA"
    assert len(model_results.warnings) == 1
    warning = model_results.warnings[0]
    assert warning.qualified_name == ("eemeter.caltrack_usage_per_day.no_data")
    assert warning.description == ("No data available. Cannot fit model.")
    assert warning.data == {}


@pytest.fixture
def cdd_hdd_h60_c65(il_electricity_cdd_hdd_daily):
    meter_data = il_electricity_cdd_hdd_daily["meter_data"]
    temperature_data = il_electricity_cdd_hdd_daily["temperature_data"]
    blackout_start_date = il_electricity_cdd_hdd_daily["blackout_start_date"]
    temperature_features = compute_temperature_features(
        meter_data.index,
        temperature_data,
        heating_balance_points=[60],
        cooling_balance_points=[65],
        use_mean_daily_values=True,
    )
    meter_data_feature = compute_usage_per_day_feature(meter_data, "meter_value")
    data = merge_features([meter_data_feature, temperature_features])
    baseline_data, warnings = get_baseline_data(data, end=blackout_start_date)
    return baseline_data


def test_fit_caltrack_usage_per_day_model_cdd_hdd(
    cdd_hdd_h60_c65, prediction_index, temperature_data
):
    model_results = fit_caltrack_usage_per_day_model(cdd_hdd_h60_c65)
    assert len(model_results.candidates) == 4
    assert model_results.candidates[0].model_type == "intercept_only"
    assert model_results.candidates[1].model_type == "hdd_only"
    assert model_results.candidates[2].model_type == "cdd_only"
    assert model_results.candidates[3].model_type == "cdd_hdd"
    assert model_results.model.status == "QUALIFIED"
    assert model_results.model.model_type == "cdd_hdd"
    model_prediction = model_results.model.predict(prediction_index, temperature_data)
    prediction_df = model_prediction.result
    assert round(prediction_df.predicted_usage.sum(), 2) == 7059.48


@pytest.fixture
def cdd_hdd_c65(il_electricity_cdd_hdd_daily):
    meter_data = il_electricity_cdd_hdd_daily["meter_data"]
    temperature_data = il_electricity_cdd_hdd_daily["temperature_data"]
    blackout_start_date = il_electricity_cdd_hdd_daily["blackout_start_date"]
    temperature_features = compute_temperature_features(
        meter_data.index,
        temperature_data,
        heating_balance_points=[],
        cooling_balance_points=[65],
        use_mean_daily_values=True,
    )
    meter_data_feature = compute_usage_per_day_feature(meter_data, "meter_value")
    data = merge_features([meter_data_feature, temperature_features])
    baseline_data, warnings = get_baseline_data(data, end=blackout_start_date)
    return baseline_data


def test_fit_caltrack_usage_per_day_model_cdd_only(
    cdd_hdd_c65, prediction_index, temperature_data
):
    model_results = fit_caltrack_usage_per_day_model(cdd_hdd_c65)
    assert len(model_results.candidates) == 2
    assert model_results.candidates[0].model_type == "intercept_only"
    assert model_results.candidates[1].model_type == "cdd_only"
    assert model_results.model.status == "QUALIFIED"
    assert model_results.model.model_type == "cdd_only"
    model_prediction = model_results.model.predict(prediction_index, temperature_data)
    prediction_df = model_prediction.result
    assert round(prediction_df.predicted_usage.sum(), 2) == 10192.0


def test_fit_caltrack_usage_per_day_model_cdd_hdd_use_billing_presets(
    cdd_hdd_h60_c65, prediction_index, temperature_data
):
    model_results = fit_caltrack_usage_per_day_model(
        cdd_hdd_h60_c65, use_billing_presets=True, weights_col="n_days_kept"
    )
    assert len(model_results.candidates) == 4
    assert model_results.candidates[0].model_type == "intercept_only"
    assert model_results.candidates[1].model_type == "hdd_only"
    assert model_results.candidates[2].model_type == "cdd_only"
    assert model_results.candidates[3].model_type == "cdd_hdd"
    assert model_results.model.status == "QUALIFIED"
    assert model_results.model.model_type == "cdd_hdd"
    model_prediction = model_results.model.predict(prediction_index, temperature_data)
    prediction = model_prediction.result
    assert round(prediction.predicted_usage.sum(), 2) == 7059.48


def test_fit_caltrack_usage_per_day_model_cdd_hdd_use_billing_presets_no_weights(
    cdd_hdd_h60_c65, prediction_index, temperature_data
):
    with pytest.raises(ValueError) as exc_info:
        fit_caltrack_usage_per_day_model(cdd_hdd_h60_c65, use_billing_presets=True)
    assert "weights_col" in str(exc_info.value)


# When model is intercept-only, num_parameters should = 0 with cvrmse = cvrmse_adj
def test_fit_caltrack_usage_per_day_model_num_parameters_equals_zero():
    data = pd.DataFrame(
        {
            "meter_value": [6, 1, 1, 6],
            "cdd_65": [5, 0, 0.1, 0],
            "hdd_65": [0, 0.1, 0.1, 5],
            "start": pd.date_range(start="2016-01-02", periods=4, freq="D", tz="UTC"),
        }
    ).set_index("start")

    model_results = fit_caltrack_usage_per_day_model(data, fit_intercept_only=True)
    assert (
        model_results.totals_metrics.num_parameters
        == model_results.totals_metrics.num_parameters
    )
    assert (
        model_results.avgs_metrics.num_parameters
        == model_results.avgs_metrics.num_parameters
    )
    assert (
        model_results.totals_metrics.cvrmse == model_results.totals_metrics.cvrmse_adj
    )
    assert model_results.avgs_metrics.cvrmse == model_results.avgs_metrics.cvrmse_adj


def test_fit_caltrack_usage_per_day_model_no_model():
    data = pd.DataFrame(
        {
            "meter_value": [4, 1, 1, 4],
            "cdd_65": [5, 0, 0.1, 0],
            "hdd_65": [0, 0.1, 0, 5],
        }
    )
    model_results = fit_caltrack_usage_per_day_model(
        data,
        fit_hdd_only=False,
        fit_cdd_hdd=False,
        fit_cdd_only=False,
        fit_intercept_only=False,
    )
    assert model_results.method_name == "caltrack_usage_per_day"
    assert model_results.status == "NO MODEL"
    assert len(model_results.warnings) == 1
    warning = model_results.warnings[0]
    assert warning.qualified_name == (
        "eemeter.caltrack_daily.select_best_candidate.no_candidates"
    )
    assert warning.description == ("No qualified model candidates available.")
    assert warning.data == {}


@pytest.fixture
def baseline_meter_data_billing():
    index = pd.date_range("2011-01-01", freq="30D", periods=12, tz="UTC")
    df = pd.DataFrame({"value": 1}, index=index)
    df.iloc[-1] = np.nan
    return df


@pytest.fixture
def baseline_temperature_data():
    index = pd.date_range("2011-01-01", freq="H", periods=1095 * 24, tz="UTC")
    series = pd.Series(np.random.normal(60, 5, len(index)), index=index)
    return series


# CalTrack 2.2.3.2
def test_caltrack_merge_temperatures_insufficient_temperature_per_period(
    baseline_meter_data_billing, baseline_temperature_data
):
    baseline_temperature_data_missing = baseline_temperature_data.copy(deep=True)
    baseline_temperature_data_missing.iloc[: (4 * 24)] = np.nan

    # test without percent_hourly_coverage_per_billing_period constraint
    temperature_features_no_constraint = compute_temperature_features(
        baseline_meter_data_billing.index,
        baseline_temperature_data_missing,
        heating_balance_points=range(40, 81),
        cooling_balance_points=range(50, 91),
        data_quality=True,
        keep_partial_nan_rows=False,
        percent_hourly_coverage_per_billing_period=0,
    )

    assert temperature_features_no_constraint["n_days_kept"].isnull().sum() == 0

    # test with default percent_hourly_coverage_per_billing_period=0.9 constraint
    temperature_features_with_constraint = compute_temperature_features(
        baseline_meter_data_billing.index,
        baseline_temperature_data_missing,
        heating_balance_points=range(40, 81),
        cooling_balance_points=range(50, 91),
        data_quality=True,
        keep_partial_nan_rows=False,
    )
    assert temperature_features_with_constraint["n_days_kept"].isnull().sum() == 1


def test_caltrack_sufficiency_criteria_no_data():
    data_quality = pd.DataFrame(
        {"meter_value": [], "temperature_not_null": [], "temperature_null": []}
    )
    data_sufficiency = caltrack_sufficiency_criteria(data_quality, None, None)
    assert data_sufficiency.status == "NO DATA"
    assert data_sufficiency.criteria_name == ("caltrack_sufficiency_criteria")
    assert len(data_sufficiency.warnings) == 1
    warning = data_sufficiency.warnings[0]
    assert warning.qualified_name == ("eemeter.caltrack_sufficiency_criteria.no_data")
    assert warning.description == "No data available."
    assert warning.data == {}


def test_caltrack_sufficiency_criteria_pass():
    data_quality = pd.DataFrame(
        {
            "meter_value": [1, np.nan],
            "temperature_not_null": [1, 1],
            "temperature_null": [0, 0],
            "start": pd.date_range(start="2016-01-02", periods=2, freq="D", tz="UTC"),
        }
    ).set_index("start")
    requested_start = pd.Timestamp("2016-01-02").tz_localize("UTC").to_pydatetime()
    requested_end = pd.Timestamp("2016-01-03").tz_localize("UTC")
    data_sufficiency = caltrack_sufficiency_criteria(
        data_quality,
        requested_start,
        requested_end,
        num_days=1,
        min_fraction_daily_coverage=0.9,
        min_fraction_hourly_temperature_coverage_per_period=0.9,
    )
    assert data_sufficiency.status == "PASS"
    assert data_sufficiency.criteria_name == ("caltrack_sufficiency_criteria")
    assert data_sufficiency.warnings == []
    assert data_sufficiency.settings == {
        "num_days": 1,
        "min_fraction_daily_coverage": 0.9,
        "min_fraction_hourly_temperature_coverage_per_period": 0.9,
    }


def test_caltrack_sufficiency_criteria_pass_extreme_value_warning():
    data_quality = pd.DataFrame(
        {
            "meter_value": [1, 1, 99999, 1, np.nan],
            "temperature_not_null": np.ones(5),
            "temperature_null": np.zeros(5),
            "start": pd.date_range(start="2016-01-02", periods=5, freq="D", tz="UTC"),
        }
    ).set_index("start")
    requested_start = pd.Timestamp("2016-01-02").tz_localize("UTC").to_pydatetime()
    requested_end = pd.Timestamp("2016-01-06").tz_localize("UTC")
    data_sufficiency = caltrack_sufficiency_criteria(
        data_quality,
        requested_start,
        requested_end,
        num_days=4,
        min_fraction_daily_coverage=0.9,
        min_fraction_hourly_temperature_coverage_per_period=0.9,
    )
    assert data_sufficiency.status == "PASS"
    assert data_sufficiency.criteria_name == ("caltrack_sufficiency_criteria")
    assert len(data_sufficiency.warnings) == 1

    warning0 = data_sufficiency.warnings[0]
    assert warning0.qualified_name == (
        "eemeter.caltrack_sufficiency_criteria.extreme_values_detected"
    )
    assert warning0.data["n_extreme_values"] == 1
    assert data_sufficiency.settings == {
        "num_days": 4,
        "min_fraction_daily_coverage": 0.9,
        "min_fraction_hourly_temperature_coverage_per_period": 0.9,
    }


def test_caltrack_sufficiency_criteria_fail_no_data():
    data_quality = pd.DataFrame(
        {
            "meter_value": [np.nan, np.nan],
            "temperature_not_null": [1, 5],
            "temperature_null": [0, 5],
            "start": pd.date_range(start="2016-01-02", periods=2, freq="D", tz="UTC"),
        }
    ).set_index("start")
    requested_start = pd.Timestamp("2016-01-02").tz_localize("UTC")
    requested_end = pd.Timestamp("2016-01-04").tz_localize("UTC")
    data_sufficiency = caltrack_sufficiency_criteria(
        data_quality,
        requested_start,
        requested_end,
        num_days=3,
        min_fraction_daily_coverage=0.9,
        min_fraction_hourly_temperature_coverage_per_period=0.9,
    )
    assert data_sufficiency.status == "NO DATA"
    assert data_sufficiency.criteria_name == ("caltrack_sufficiency_criteria")


def test_caltrack_sufficiency_criteria_fail_no_gap():
    data_quality = pd.DataFrame(
        {
            "meter_value": [1, 1],
            "temperature_not_null": [1, 5],
            "temperature_null": [0, 5],
            "start": pd.date_range(start="2016-01-02", periods=2, freq="D", tz="UTC"),
        }
    ).set_index("start")
    requested_start = pd.Timestamp("2016-01-02").tz_localize("UTC")
    requested_end = pd.Timestamp("2016-01-04").tz_localize("UTC")
    data_sufficiency = caltrack_sufficiency_criteria(
        data_quality,
        requested_start,
        requested_end,
        num_days=3,
        min_fraction_daily_coverage=0.9,
        min_fraction_hourly_temperature_coverage_per_period=0.9,
    )
    assert data_sufficiency.status == "FAIL"
    assert data_sufficiency.criteria_name == ("caltrack_sufficiency_criteria")
    assert len(data_sufficiency.warnings) == 4

    warning0 = data_sufficiency.warnings[0]
    assert warning0.qualified_name == (
        "eemeter.caltrack_sufficiency_criteria.incorrect_number_of_total_days"
    )
    assert warning0.description == (
        "Total data span does not match the required value."
    )
    assert warning0.data == {"num_days": 3, "n_days_total": 2}

    warning1 = data_sufficiency.warnings[1]
    assert warning1.qualified_name == (
        "eemeter.caltrack_sufficiency_criteria." "too_many_days_with_missing_data"
    )
    assert warning1.description == (
        "Too many days in data have missing meter data or temperature data."
    )
    assert warning1.data == {"n_days_total": 2, "n_valid_days": 1}

    warning2 = data_sufficiency.warnings[2]
    assert warning2.qualified_name == (
        "eemeter.caltrack_sufficiency_criteria." "too_many_days_with_missing_meter_data"
    )
    assert warning2.description == ("Too many days in data have missing meter data.")
    # zero because nan value and last point dropped
    assert warning2.data == {"n_days_total": 2, "n_valid_meter_data_days": 1}

    warning3 = data_sufficiency.warnings[3]
    assert warning3.qualified_name == (
        "eemeter.caltrack_sufficiency_criteria."
        "too_many_days_with_missing_temperature_data"
    )
    assert warning3.description == (
        "Too many days in data have missing temperature data."
    )
    assert warning3.data == {"n_days_total": 2, "n_valid_temperature_data_days": 1}


def test_caltrack_sufficiency_criteria_pass_no_requested_start_end():
    data_quality = pd.DataFrame(
        {
            "meter_value": [1, np.nan],
            "temperature_not_null": [1, 1],
            "temperature_null": [0, 0],
            "start": pd.date_range(start="2016-01-02", periods=2, freq="D", tz="UTC"),
        }
    ).set_index("start")
    data_sufficiency = caltrack_sufficiency_criteria(
        data_quality,
        None,
        None,
        num_days=1,
        min_fraction_daily_coverage=0.9,
        min_fraction_hourly_temperature_coverage_per_period=0.9,
    )
    assert data_sufficiency.status == "PASS"
    assert data_sufficiency.criteria_name == ("caltrack_sufficiency_criteria")
    assert len(data_sufficiency.warnings) == 0


def test_caltrack_sufficiency_criteria_fail_with_requested_start_end():
    data_quality = pd.DataFrame(
        {
            "meter_value": [1, np.nan],
            "temperature_not_null": [1, 1],
            "temperature_null": [0, 0],
            "start": pd.date_range(start="2016-01-02", periods=2, freq="D", tz="UTC"),
        }
    ).set_index("start")
    requested_start = pd.Timestamp("2016-01-01").tz_localize("UTC")
    requested_end = pd.Timestamp("2016-01-04").tz_localize("UTC")
    data_sufficiency = caltrack_sufficiency_criteria(
        data_quality,
        requested_start,
        requested_end,
        num_days=3,
        min_fraction_daily_coverage=0.9,
        min_fraction_hourly_temperature_coverage_per_period=0.9,
    )
    assert data_sufficiency.status == "FAIL"
    assert data_sufficiency.criteria_name == ("caltrack_sufficiency_criteria")
    assert len(data_sufficiency.warnings) == 3

    warning0 = data_sufficiency.warnings[0]
    assert warning0.qualified_name == (
        "eemeter.caltrack_sufficiency_criteria." "too_many_days_with_missing_data"
    )
    assert warning0.description == (
        "Too many days in data have missing meter data or temperature data."
    )
    assert warning0.data == {"n_days_total": 3, "n_valid_days": 1}

    warning1 = data_sufficiency.warnings[1]
    assert warning1.qualified_name == (
        "eemeter.caltrack_sufficiency_criteria." "too_many_days_with_missing_meter_data"
    )
    assert warning1.description == ("Too many days in data have missing meter data.")
    assert warning1.data == {"n_days_total": 3, "n_valid_meter_data_days": 1}

    warning2 = data_sufficiency.warnings[2]
    assert warning2.qualified_name == (
        "eemeter.caltrack_sufficiency_criteria."
        "too_many_days_with_missing_temperature_data"
    )
    assert warning2.description == (
        "Too many days in data have missing temperature data."
    )
    assert warning2.data == {"n_days_total": 3, "n_valid_temperature_data_days": 1}


# CalTrack 2.2.4
def test_caltrack_sufficiency_criteria_too_much_data():
    data_quality = pd.DataFrame(
        {
            "meter_value": [1, 1, np.nan],
            "temperature_not_null": [1, 1, 1],
            "temperature_null": [0, 0, 0],
            "start": pd.date_range(start="2016-01-02", periods=3, freq="D", tz="UTC"),
        }
    ).set_index("start")
    requested_start = pd.Timestamp("2016-01-03").tz_localize("UTC")
    requested_end = pd.Timestamp("2016-01-03").tz_localize("UTC")
    data_sufficiency = caltrack_sufficiency_criteria(
        data_quality,
        requested_start,
        requested_end,
        num_days=2,
        min_fraction_daily_coverage=0.9,
        min_fraction_hourly_temperature_coverage_per_period=0.9,
    )
    assert data_sufficiency.status == "FAIL"
    assert len(data_sufficiency.warnings) == 2

    warning0 = data_sufficiency.warnings[0]
    assert warning0.qualified_name == (
        "eemeter.caltrack_sufficiency_criteria." "extra_data_after_requested_end_date"
    )
    assert warning0.description == ("Extra data found after requested end date.")
    assert warning0.data == {
        "data_end": "2016-01-04T00:00:00+00:00",
        "requested_end": "2016-01-03T00:00:00+00:00",
    }

    warning1 = data_sufficiency.warnings[1]
    assert warning1.qualified_name == (
        "eemeter.caltrack_sufficiency_criteria."
        "extra_data_before_requested_start_date"
    )
    assert warning1.description == ("Extra data found before requested start date.")
    assert warning1.data == {
        "data_start": "2016-01-02T00:00:00+00:00",
        "requested_start": "2016-01-03T00:00:00+00:00",
    }


def test_caltrack_sufficiency_criteria_negative_values():
    data_quality = pd.DataFrame(
        {
            "meter_value": [-1, 1, np.nan],
            "temperature_not_null": [1, 1, 1],
            "temperature_null": [0, 0, 1],
            "start": pd.date_range(start="2016-01-02", periods=3, freq="D", tz="UTC"),
        }
    ).set_index("start")
    data_sufficiency = caltrack_sufficiency_criteria(
        data_quality,
        None,
        None,
        num_days=2,
        min_fraction_daily_coverage=0.9,
        min_fraction_hourly_temperature_coverage_per_period=0.9,
    )
    assert data_sufficiency.status == "FAIL"
    assert len(data_sufficiency.warnings) == 1

    warning0 = data_sufficiency.warnings[0]
    assert warning0.qualified_name == (
        "eemeter.caltrack_sufficiency_criteria.negative_meter_values"
    )
    assert warning0.description == (
        "Found negative meter data values, which may indicate presence of"
        " solar net metering."
    )
    assert warning0.data == {"n_negative_meter_values": 1}


def test_caltrack_sufficiency_criteria_handle_single_input():
    # just make sure this case is handled.
    data_quality = pd.DataFrame(
        {
            "meter_value": [1],
            "temperature_not_null": [1],
            "temperature_null": [0],
            "start": pd.date_range(start="2016-01-02", periods=1, freq="D", tz="UTC"),
        }
    ).set_index("start")
    data_sufficiency = caltrack_sufficiency_criteria(
        data_quality,
        None,
        None,
        num_days=0,
        min_fraction_daily_coverage=0.9,
        min_fraction_hourly_temperature_coverage_per_period=0.9,
    )

    assert data_sufficiency.status == "FAIL"
    assert len(data_sufficiency.warnings) == 3


def test_caltrack_usage_per_day_predict_empty(prediction_index, temperature_data):
    prediction = caltrack_usage_per_day_predict(
        "intercept_only",
        {"intercept": 1},
        prediction_index[:0],
        temperature_data,
        with_disaggregated=True,
        with_design_matrix=True,
    )
    assert sorted(prediction.result.columns) == [
        "base_load",
        "cooling_load",
        "heating_load",
        "predicted_usage",
    ]
    assert round(prediction.result.predicted_usage.sum(), 2) == 0


def test_caltrack_usage_per_day_temp_empty(prediction_index, temperature_data):
    prediction = caltrack_usage_per_day_predict(
        "intercept_only",
        {"intercept": 1},
        prediction_index,
        temperature_data[:0],
        with_disaggregated=True,
        with_design_matrix=True,
    )
    assert sorted(prediction.result.columns) == [
        "base_load",
        "cooling_load",
        "heating_load",
        "predicted_usage",
    ]
    assert round(prediction.result.predicted_usage.sum(), 2) == 0
    assert prediction.result.predicted_usage.shape[0] == 0
