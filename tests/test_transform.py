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
from datetime import datetime, timedelta
from pkg_resources import resource_stream

import pandas as pd
import pytest

from eemeter.transform import (
    as_freq,
    day_counts,
    get_baseline_data,
    get_reporting_data,
    remove_duplicates,
    NoBaselineDataError,
    NoReportingDataError,
)


def test_as_freq_not_series(il_electricity_cdd_hdd_billing_monthly):
    meter_data = il_electricity_cdd_hdd_billing_monthly["meter_data"]
    assert meter_data.shape == (27, 1)
    with pytest.raises(ValueError):
        as_freq(meter_data, freq="H")


def test_as_freq_hourly(il_electricity_cdd_hdd_billing_monthly):
    meter_data = il_electricity_cdd_hdd_billing_monthly["meter_data"]
    assert meter_data.shape == (27, 1)
    as_hourly = as_freq(meter_data.value, freq="H")
    assert as_hourly.shape == (18961,)
    assert round(meter_data.value.sum(), 1) == round(as_hourly.sum(), 1) == 21290.2


def test_as_freq_daily(il_electricity_cdd_hdd_billing_monthly):
    meter_data = il_electricity_cdd_hdd_billing_monthly["meter_data"]
    assert meter_data.shape == (27, 1)
    as_daily = as_freq(meter_data.value, freq="D")
    assert as_daily.shape == (791,)
    assert round(meter_data.value.sum(), 1) == round(as_daily.sum(), 1) == 21290.2


def test_as_freq_month_start(il_electricity_cdd_hdd_billing_monthly):
    meter_data = il_electricity_cdd_hdd_billing_monthly["meter_data"]
    assert meter_data.shape == (27, 1)
    as_month_start = as_freq(meter_data.value, freq="MS")
    assert as_month_start.shape == (27,)
    assert round(meter_data.value.sum(), 1) == round(as_month_start.sum(), 1) == 21290.2


def test_as_freq_hourly_temperature(il_electricity_cdd_hdd_billing_monthly):
    temperature_data = il_electricity_cdd_hdd_billing_monthly["temperature_data"]
    assert temperature_data.shape == (19417, )
    as_hourly = as_freq(temperature_data, freq="H", series_type='instantaneous')
    assert as_hourly.shape == (19417,)
    assert round(temperature_data.mean(), 1) == round(as_hourly.mean(), 1) == 54.6


def test_as_freq_daily_temperature(il_electricity_cdd_hdd_billing_monthly):
    temperature_data = il_electricity_cdd_hdd_billing_monthly["temperature_data"]
    assert temperature_data.shape == (19417, )
    as_daily = as_freq(temperature_data, freq="D", series_type='instantaneous')
    assert as_daily.shape == (810,)
    assert abs(temperature_data.mean() - as_daily.mean()) <= 0.1


def test_as_freq_month_start_temperature(il_electricity_cdd_hdd_billing_monthly):
    temperature_data = il_electricity_cdd_hdd_billing_monthly["temperature_data"]
    assert temperature_data.shape == (19417, )
    as_month_start = as_freq(temperature_data, freq="MS", series_type='instantaneous')
    assert as_month_start.shape == (28,)
    assert round(as_month_start.mean(), 1) == 53.4


def test_as_freq_daily_temperature_monthly(il_electricity_cdd_hdd_billing_monthly):
    temperature_data = il_electricity_cdd_hdd_billing_monthly["temperature_data"]
    temperature_data = temperature_data.groupby(pd.Grouper(freq='MS')).mean()
    assert temperature_data.shape == (28, )
    as_daily = as_freq(temperature_data, freq="D", series_type='instantaneous')
    assert as_daily.shape == (824,)
    assert round(as_daily.mean(), 1) == 54.5


def test_as_freq_empty():
    meter_data = pd.DataFrame({"value": []})
    empty_meter_data = as_freq(meter_data.value, freq="H")
    assert empty_meter_data.empty


def test_day_counts(il_electricity_cdd_hdd_billing_monthly):
    data = il_electricity_cdd_hdd_billing_monthly["meter_data"].value
    counts = day_counts(data.index)
    assert counts.shape == (27,)
    assert counts.iloc[0] == 29.0
    assert pd.isnull(counts.iloc[-1])
    assert counts.sum() == 790.0


def test_day_counts_empty_series():
    index = pd.DatetimeIndex([])
    index.freq = None
    data = pd.Series([], index=index)
    counts = day_counts(data.index)
    assert counts.shape == (0,)


def test_get_baseline_data(il_electricity_cdd_hdd_hourly):
    meter_data = il_electricity_cdd_hdd_hourly["meter_data"]
    baseline_data, warnings = get_baseline_data(meter_data)
    assert meter_data.shape == baseline_data.shape == (19417, 1)
    assert len(warnings) == 0


def test_get_baseline_data_with_end(il_electricity_cdd_hdd_hourly):
    meter_data = il_electricity_cdd_hdd_hourly["meter_data"]
    blackout_start_date = il_electricity_cdd_hdd_hourly["blackout_start_date"]
    baseline_data, warnings = get_baseline_data(meter_data, end=blackout_start_date)
    assert meter_data.shape != baseline_data.shape == (8761, 1)
    assert len(warnings) == 0


def test_get_baseline_data_with_end_no_max_days(il_electricity_cdd_hdd_hourly):
    meter_data = il_electricity_cdd_hdd_hourly["meter_data"]
    blackout_start_date = il_electricity_cdd_hdd_hourly["blackout_start_date"]
    baseline_data, warnings = get_baseline_data(
        meter_data, end=blackout_start_date, max_days=None
    )
    assert meter_data.shape != baseline_data.shape == (9595, 1)
    assert len(warnings) == 0


def test_get_baseline_data_empty(il_electricity_cdd_hdd_hourly):
    meter_data = il_electricity_cdd_hdd_hourly["meter_data"]
    blackout_start_date = il_electricity_cdd_hdd_hourly["blackout_start_date"]
    with pytest.raises(NoBaselineDataError):
        get_baseline_data(meter_data, end=pd.Timestamp("2000").tz_localize("UTC"))


def test_get_baseline_data_start_gap(il_electricity_cdd_hdd_hourly):
    meter_data = il_electricity_cdd_hdd_hourly["meter_data"]
    start = meter_data.index.min() - timedelta(days=1)
    baseline_data, warnings = get_baseline_data(meter_data, start=start)
    assert meter_data.shape == baseline_data.shape == (19417, 1)
    assert len(warnings) == 1
    warning = warnings[0]
    assert warning.qualified_name == "eemeter.get_baseline_data.gap_at_baseline_start"
    assert (
        warning.description
        == "Data does not have coverage at requested baseline start date."
    )
    assert warning.data == {
        "data_start": "2015-11-22T06:00:00+00:00",
        "requested_start": "2015-11-21T06:00:00+00:00",
    }


def test_get_baseline_data_end_gap(il_electricity_cdd_hdd_hourly):
    meter_data = il_electricity_cdd_hdd_hourly["meter_data"]
    end = meter_data.index.max() + timedelta(days=1)
    baseline_data, warnings = get_baseline_data(meter_data, end=end, max_days=None)
    assert meter_data.shape == baseline_data.shape == (19417, 1)
    assert len(warnings) == 1
    warning = warnings[0]
    assert warning.qualified_name == "eemeter.get_baseline_data.gap_at_baseline_end"
    assert (
        warning.description
        == "Data does not have coverage at requested baseline end date."
    )
    assert warning.data == {
        "data_end": "2018-02-08T06:00:00+00:00",
        "requested_end": "2018-02-09T06:00:00+00:00",
    }


def test_get_reporting_data(il_electricity_cdd_hdd_hourly):
    meter_data = il_electricity_cdd_hdd_hourly["meter_data"]
    reporting_data, warnings = get_reporting_data(meter_data)
    assert meter_data.shape == reporting_data.shape == (19417, 1)
    assert len(warnings) == 0


def test_get_reporting_data_with_start(il_electricity_cdd_hdd_hourly):
    meter_data = il_electricity_cdd_hdd_hourly["meter_data"]
    blackout_end_date = il_electricity_cdd_hdd_hourly["blackout_end_date"]
    reporting_data, warnings = get_reporting_data(meter_data, start=blackout_end_date)
    assert meter_data.shape != reporting_data.shape == (8761, 1)
    assert len(warnings) == 0


def test_get_reporting_data_with_start_no_max_days(il_electricity_cdd_hdd_hourly):
    meter_data = il_electricity_cdd_hdd_hourly["meter_data"]
    blackout_end_date = il_electricity_cdd_hdd_hourly["blackout_end_date"]
    reporting_data, warnings = get_reporting_data(
        meter_data, start=blackout_end_date, max_days=None
    )
    assert meter_data.shape != reporting_data.shape == (9607, 1)
    assert len(warnings) == 0


def test_get_reporting_data_empty(il_electricity_cdd_hdd_hourly):
    meter_data = il_electricity_cdd_hdd_hourly["meter_data"]
    blackout_end_date = il_electricity_cdd_hdd_hourly["blackout_end_date"]
    with pytest.raises(NoReportingDataError):
        get_reporting_data(meter_data, start=pd.Timestamp("2030").tz_localize("UTC"))


def test_get_reporting_data_start_gap(il_electricity_cdd_hdd_hourly):
    meter_data = il_electricity_cdd_hdd_hourly["meter_data"]
    start = meter_data.index.min() - timedelta(days=1)
    reporting_data, warnings = get_reporting_data(
        meter_data, start=start, max_days=None
    )
    assert meter_data.shape == reporting_data.shape == (19417, 1)
    assert len(warnings) == 1
    warning = warnings[0]
    assert warning.qualified_name == "eemeter.get_reporting_data.gap_at_reporting_start"
    assert (
        warning.description
        == "Data does not have coverage at requested reporting start date."
    )
    assert warning.data == {
        "data_start": "2015-11-22T06:00:00+00:00",
        "requested_start": "2015-11-21T06:00:00+00:00",
    }


def test_get_reporting_data_end_gap(il_electricity_cdd_hdd_hourly):
    meter_data = il_electricity_cdd_hdd_hourly["meter_data"]
    end = meter_data.index.max() + timedelta(days=1)
    reporting_data, warnings = get_reporting_data(meter_data, end=end)
    assert meter_data.shape == reporting_data.shape == (19417, 1)
    assert len(warnings) == 1
    warning = warnings[0]
    assert warning.qualified_name == "eemeter.get_reporting_data.gap_at_reporting_end"
    assert (
        warning.description
        == "Data does not have coverage at requested reporting end date."
    )
    assert warning.data == {
        "data_end": "2018-02-08T06:00:00+00:00",
        "requested_end": "2018-02-09T06:00:00+00:00",
    }


def test_remove_duplicates_df():
    index = pd.DatetimeIndex(["2017-01-01", "2017-01-02", "2017-01-02"])
    df = pd.DataFrame({"value": [1, 2, 3]}, index=index)
    assert df.shape == (3, 1)
    df_dedupe = remove_duplicates(df)
    assert df_dedupe.shape == (2, 1)
    assert list(df_dedupe.value) == [1, 2]


def test_remove_duplicates_series():
    index = pd.DatetimeIndex(["2017-01-01", "2017-01-02", "2017-01-02"])
    series = pd.Series([1, 2, 3], index=index)
    assert series.shape == (3,)
    series_dedupe = remove_duplicates(series)
    assert series_dedupe.shape == (2,)
    assert list(series_dedupe) == [1, 2]
