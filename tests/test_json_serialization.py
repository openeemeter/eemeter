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
from eemeter.eemeter.samples import load_sample
from eemeter.eemeter.common.transform import get_baseline_data, get_reporting_data
from eemeter.eemeter import (
    DailyBaselineData,
    DailyModel,
    BillingBaselineData,
    BillingReportingData,
    BillingModel,
    HourlyModel,
    HourlyBaselineData,
    HourlyReportingData,
)



def test_json_daily():
    meter_data, temperature_data, sample_metadata = load_sample(
        "il-electricity-cdd-hdd-daily"
    )

    blackout_start_date = sample_metadata["blackout_start_date"]
    blackout_end_date = sample_metadata["blackout_end_date"]

    # fit baseline model
    baseline_meter_data, warnings = get_baseline_data(
        meter_data, end=blackout_start_date, max_days=365
    )
    baseline_data = DailyBaselineData.from_series(
        baseline_meter_data, temperature_data, is_electricity_data=True
    )
    baseline_model = DailyModel().fit(baseline_data, ignore_disqualification=True)

    # predict on reporting year and calculate savings
    reporting_meter_data, warnings = get_reporting_data(
        meter_data, start=blackout_end_date, max_days=365
    )
    # TODO change to Reporting once class is fixed
    reporting_data = DailyBaselineData.from_series(
        reporting_meter_data, temperature_data, is_electricity_data=True
    )
    metered_savings_dataframe = baseline_model.predict(reporting_data)
    total_metered_savings = (
        metered_savings_dataframe["observed"] - metered_savings_dataframe["predicted"]
    ).sum()

    # serialize, deserialize model
    json_str = baseline_model.to_json()
    loaded_model = DailyModel.from_json(json_str)

    # compute metered savings from the loaded model
    prediction_json = loaded_model.predict(reporting_data)
    total_metered_savings_loaded = (
        prediction_json["observed"] - prediction_json["predicted"]
    ).sum()

    # compare results
    assert total_metered_savings == total_metered_savings_loaded


def test_json_billing():
    meter_data, temperature_data, sample_metadata = load_sample(
        "il-electricity-cdd-hdd-billing_monthly"
    )

    blackout_start_date = sample_metadata["blackout_start_date"]
    blackout_end_date = sample_metadata["blackout_end_date"]

    # fit baseline model
    baseline_meter_data, warnings = get_baseline_data(
        meter_data, end=blackout_start_date, max_days=365
    )
    baseline_data = BillingBaselineData.from_series(
        baseline_meter_data, temperature_data, is_electricity_data=True
    )
    baseline_model = BillingModel().fit(baseline_data, ignore_disqualification=True)

    # predict on reporting year and calculate savings
    reporting_meter_data, warnings = get_reporting_data(
        meter_data, start=blackout_end_date, max_days=365
    )
    reporting_data = BillingReportingData.from_series(
        reporting_meter_data, temperature_data, is_electricity_data=True
    )
    metered_savings_dataframe = baseline_model.predict(reporting_data)
    total_metered_savings = (
        metered_savings_dataframe["observed"] - metered_savings_dataframe["predicted"]
    ).sum()

    # serialize, deserialize model
    json_str = baseline_model.to_json()
    loaded_model = BillingModel.from_json(json_str)

    # compute metered savings from the loaded model
    prediction_json = loaded_model.predict(reporting_data)
    total_metered_savings_loaded = (
        prediction_json["observed"] - prediction_json["predicted"]
    ).sum()

    # compare results
    assert total_metered_savings == total_metered_savings_loaded


def test_json_hourly():
    meter_data, temperature_data, sample_metadata = load_sample(
        "il-electricity-cdd-hdd-hourly"
    )

    blackout_start_date = sample_metadata["blackout_start_date"]
    blackout_end_date = sample_metadata["blackout_end_date"]

    # get meter data suitable for fitting a baseline model
    baseline_meter_data, warnings = get_baseline_data(
        meter_data, end=blackout_start_date, max_days=365
    )
    baseline = HourlyBaselineData.from_series(
        baseline_meter_data, temperature_data, is_electricity_data=True
    )

    # build a CalTRACK hourly model
    baseline_model = HourlyModel().fit(baseline)

    # get a year of reporting period data
    reporting_meter_data, warnings = get_reporting_data(
        meter_data, start=blackout_end_date, max_days=365
    )
    reporting = HourlyReportingData.from_series(
        reporting_meter_data, temperature_data, is_electricity_data=True
    )

    result1 = baseline_model.predict(reporting)

    # serialize, deserialize
    json_str = baseline_model.to_json()
    m = HourlyModel.from_json(json_str)

    result2 = m.predict(reporting)

    assert result1["predicted"].sum() == result2["predicted"].sum()
