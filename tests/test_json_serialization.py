#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

   Copyright 2014-2020 OpenEEmeter contributors

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
import eemeter
from eemeter import DailyBaselineData, DailyReportingData, DailyModel, BillingBaselineData, BillingReportingData, BillingModel
import json


def test_json_daily():
    meter_data, temperature_data, sample_metadata = eemeter.load_sample(
        "il-electricity-cdd-hdd-daily"
    )

    blackout_start_date = sample_metadata["blackout_start_date"]
    blackout_end_date = sample_metadata["blackout_end_date"]

    # fit baseline model
    baseline_meter_data, warnings = eemeter.get_baseline_data(
        meter_data, end=blackout_start_date, max_days=365
    )
    baseline_data = DailyBaselineData.from_series(baseline_meter_data, temperature_data, is_electricity_data=True)
    baseline_model = DailyModel().fit(baseline_data, ignore_disqualification=True)

    # predict on reporting year and calculate savings
    reporting_meter_data, warnings = eemeter.get_reporting_data(
        meter_data, start=blackout_end_date, max_days=365
    )
    #TODO change to Reporting once class is fixed
    reporting_data = DailyBaselineData.from_series(reporting_meter_data, temperature_data, is_electricity_data=True)
    metered_savings_dataframe = baseline_model.predict(reporting_data)
    total_metered_savings = (metered_savings_dataframe['observed'] - metered_savings_dataframe['predicted']).sum()

    # serialize, deserialize model
    json_str = baseline_model.to_json()
    loaded_model = DailyModel.from_json(json_str)

    # compute metered savings from the loaded model
    prediction_json = loaded_model.predict(reporting_data)
    total_metered_savings_loaded = (prediction_json['observed'] - prediction_json['predicted']).sum()

    # compare results
    assert total_metered_savings == total_metered_savings_loaded

def test_json_billing():
    meter_data, temperature_data, sample_metadata = eemeter.load_sample(
        "il-electricity-cdd-hdd-billing_monthly"
    )

    blackout_start_date = sample_metadata["blackout_start_date"]
    blackout_end_date = sample_metadata["blackout_end_date"]

    # fit baseline model
    baseline_meter_data, warnings = eemeter.get_baseline_data(
        meter_data, end=blackout_start_date, max_days=365
    )
    baseline_data = BillingBaselineData.from_series(baseline_meter_data, temperature_data, is_electricity_data=True)
    baseline_model = BillingModel().fit(baseline_data, ignore_disqualification=True)

    # predict on reporting year and calculate savings
    reporting_meter_data, warnings = eemeter.get_reporting_data(
        meter_data, start=blackout_end_date, max_days=365
    )
    reporting_data = BillingReportingData.from_series(reporting_meter_data, temperature_data, is_electricity_data=True)
    metered_savings_dataframe = baseline_model.predict(reporting_data)
    total_metered_savings = (metered_savings_dataframe['observed'] - metered_savings_dataframe['predicted']).sum()

    # serialize, deserialize model
    json_str = baseline_model.to_json()
    loaded_model = BillingModel.from_json(json_str)

    # compute metered savings from the loaded model
    prediction_json = loaded_model.predict(reporting_data)
    total_metered_savings_loaded = (prediction_json['observed'] - prediction_json['predicted']).sum()

    # compare results
    assert total_metered_savings == total_metered_savings_loaded


def test_json_hourly():
    meter_data, temperature_data, sample_metadata = eemeter.load_sample(
        "il-electricity-cdd-hdd-hourly"
    )

    blackout_start_date = sample_metadata["blackout_start_date"]
    blackout_end_date = sample_metadata["blackout_end_date"]

    # get meter data suitable for fitting a baseline model
    baseline_meter_data, warnings = eemeter.get_baseline_data(
        meter_data, end=blackout_start_date, max_days=365
    )

    # create a design matrix for occupancy and segmentation
    preliminary_design_matrix = (
        eemeter.create_caltrack_hourly_preliminary_design_matrix(
            baseline_meter_data, temperature_data
        )
    )

    # build 12 monthly models - each step from now on operates on each segment
    segmentation = eemeter.segment_time_series(
        preliminary_design_matrix.index, "three_month_weighted"
    )

    # assign an occupancy status to each hour of the week (0-167)
    occupancy_lookup = eemeter.estimate_hour_of_week_occupancy(
        preliminary_design_matrix, segmentation=segmentation
    )

    # assign temperatures to bins
    (
        occupied_temperature_bins,
        unoccupied_temperature_bins,
    ) = eemeter.fit_temperature_bins(
        preliminary_design_matrix,
        segmentation=segmentation,
        occupancy_lookup=occupancy_lookup,
    )

    # build a design matrix for each monthly segment
    segmented_design_matrices = (
        eemeter.create_caltrack_hourly_segmented_design_matrices(
            preliminary_design_matrix,
            segmentation,
            occupancy_lookup,
            occupied_temperature_bins,
            unoccupied_temperature_bins,
        )
    )

    # build a CalTRACK hourly model
    baseline_model = eemeter.fit_caltrack_hourly_model(
        segmented_design_matrices,
        occupancy_lookup,
        occupied_temperature_bins,
        unoccupied_temperature_bins,
    )

    # get a year of reporting period data
    reporting_meter_data, warnings = eemeter.get_reporting_data(
        meter_data, start=blackout_end_date, max_days=365
    )

    # compute metered savings
    metered_savings_dataframe, error_bands = eemeter.metered_savings(
        baseline_model, reporting_meter_data, temperature_data, with_disaggregated=True
    )

    # total metered savings
    total_metered_savings = metered_savings_dataframe.metered_savings.sum()

    # test JSON
    json_str = json.dumps(baseline_model.json())

    m = eemeter.CalTRACKHourlyModelResults.from_json(json.loads(json_str))

    # compute metered savings from the loaded model
    metered_savings_dataframe, error_bands = eemeter.metered_savings(
        m, reporting_meter_data, temperature_data, with_disaggregated=True
    )

    # total metered savings
    total_metered_savings_2 = metered_savings_dataframe.metered_savings.sum()

    assert total_metered_savings == total_metered_savings_2
