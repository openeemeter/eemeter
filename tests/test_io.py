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
import gzip
from pkg_resources import resource_filename, resource_stream
from tempfile import TemporaryFile

import pandas as pd
import pytest

from eemeter import (
    meter_data_from_csv,
    meter_data_from_json,
    meter_data_to_csv,
    temperature_data_from_csv,
    temperature_data_from_json,
    temperature_data_to_csv,
)


def test_meter_data_from_csv(sample_metadata):
    meter_item = sample_metadata["il-electricity-cdd-hdd-daily"]
    meter_data_filename = meter_item["meter_data_filename"]

    fname = resource_filename("eemeter.samples", meter_data_filename)
    with gzip.open(fname) as f:
        meter_data = meter_data_from_csv(f)
    assert meter_data.shape == (810, 1)
    assert meter_data.index.tz.zone == "UTC"
    assert meter_data.index.freq is None


def test_meter_data_from_csv_gzipped(sample_metadata):
    meter_item = sample_metadata["il-electricity-cdd-hdd-daily"]
    meter_data_filename = meter_item["meter_data_filename"]

    with resource_stream("eemeter.samples", meter_data_filename) as f:
        meter_data = meter_data_from_csv(f, gzipped=True)
    assert meter_data.shape == (810, 1)
    assert meter_data.index.tz.zone == "UTC"
    assert meter_data.index.freq is None


def test_meter_data_from_csv_with_tz(sample_metadata):
    meter_item = sample_metadata["il-electricity-cdd-hdd-daily"]
    meter_data_filename = meter_item["meter_data_filename"]

    with resource_stream("eemeter.samples", meter_data_filename) as f:
        meter_data = meter_data_from_csv(f, gzipped=True, tz="US/Eastern")
    assert meter_data.shape == (810, 1)
    assert meter_data.index.tz.zone == "US/Eastern"
    assert meter_data.index.freq is None


def test_meter_data_from_csv_hourly_freq(sample_metadata):
    meter_item = sample_metadata["il-electricity-cdd-hdd-daily"]
    meter_data_filename = meter_item["meter_data_filename"]

    with resource_stream("eemeter.samples", meter_data_filename) as f:
        meter_data = meter_data_from_csv(f, gzipped=True, freq="hourly")
    assert meter_data.shape == (19417, 1)
    assert meter_data.index.tz.zone == "UTC"
    assert meter_data.index.freq == "H"


def test_meter_data_from_csv_daily_freq(sample_metadata):
    meter_item = sample_metadata["il-electricity-cdd-hdd-daily"]
    meter_data_filename = meter_item["meter_data_filename"]

    with resource_stream("eemeter.samples", meter_data_filename) as f:
        meter_data = meter_data_from_csv(f, gzipped=True, freq="daily")
    assert meter_data.shape == (810, 1)
    assert meter_data.index.tz.zone == "UTC"
    assert meter_data.index.freq == "D"


def test_meter_data_from_csv_custom_columns(sample_metadata):
    with TemporaryFile() as f:
        f.write(b"start_custom,kWh\n" b"2017-01-01T00:00:00,10\n")
        f.seek(0)
        meter_data = meter_data_from_csv(f, start_col="start_custom", value_col="kWh")
    assert meter_data.shape == (1, 1)
    assert meter_data.index.tz.zone == "UTC"
    assert meter_data.index.freq is None


def test_meter_data_from_json_orient_list(sample_metadata):
    data = [["2017-01-01T00:00:00Z", 11], ["2017-01-02T00:00:00Z", 10]]
    meter_data = meter_data_from_json(data, orient="list")
    assert meter_data.shape == (2, 1)
    assert meter_data.index.tz.zone == "UTC"
    assert meter_data.index.freq is None


def test_meter_data_from_json_bad_orient(sample_metadata):
    data = [["2017-01-01T00:00:00Z", 11], ["2017-01-02T00:00:00Z", 10]]
    with pytest.raises(ValueError):
        meter_data_from_json(data, orient="NOT_ALLOWED")


def test_meter_data_to_csv(sample_metadata):
    df = pd.DataFrame({"value": [5]}, index=pd.DatetimeIndex(["2017-01-01T00:00:00Z"]))
    with TemporaryFile("w+") as f:
        meter_data_to_csv(df, f)
        f.seek(0)
        assert f.read() == ("start,value\n" "2017-01-01,5\n")


def test_temperature_data_from_csv(sample_metadata):
    meter_item = sample_metadata["il-electricity-cdd-hdd-daily"]
    temperature_filename = meter_item["temperature_filename"]

    fname = resource_filename("eemeter.samples", temperature_filename)
    with gzip.open(fname) as f:
        temperature_data = temperature_data_from_csv(f)
    assert temperature_data.shape == (19417,)
    assert temperature_data.index.tz.zone == "UTC"
    assert temperature_data.index.freq is None


def test_temperature_data_from_csv_gzipped(sample_metadata):
    meter_item = sample_metadata["il-electricity-cdd-hdd-daily"]
    temperature_filename = meter_item["temperature_filename"]

    with resource_stream("eemeter.samples", temperature_filename) as f:
        temperature_data = temperature_data_from_csv(f, gzipped=True)
    assert temperature_data.shape == (19417,)
    assert temperature_data.index.tz.zone == "UTC"
    assert temperature_data.index.freq is None


def test_temperature_data_from_csv_hourly_freq(sample_metadata):
    meter_item = sample_metadata["il-electricity-cdd-hdd-daily"]
    temperature_filename = meter_item["temperature_filename"]

    with resource_stream("eemeter.samples", temperature_filename) as f:
        temperature_data = temperature_data_from_csv(f, gzipped=True, freq="hourly")
    assert temperature_data.shape == (19417,)
    assert temperature_data.index.tz.zone == "UTC"
    assert temperature_data.index.freq == "H"


def test_temperature_data_from_csv_custom_columns(sample_metadata):
    with TemporaryFile() as f:
        f.write(b"dt_custom,tempC\n" b"2017-01-01T00:00:00,10\n")
        f.seek(0)
        temperature_data = temperature_data_from_csv(
            f, date_col="dt_custom", temp_col="tempC"
        )
    assert temperature_data.shape == (1,)
    assert temperature_data.index.tz.zone == "UTC"
    assert temperature_data.index.freq is None


def test_temperature_data_from_json_orient_list(sample_metadata):
    data = [["2017-01-01T00:00:00Z", 11], ["2017-01-02T00:00:00Z", 10]]
    temperature_data = temperature_data_from_json(data, orient="list")
    assert temperature_data.shape == (2,)
    assert temperature_data.index.tz.zone == "UTC"
    assert temperature_data.index.freq is None


def test_temperature_data_from_json_bad_orient(sample_metadata):
    data = [["2017-01-01T00:00:00Z", 11], ["2017-01-02T00:00:00Z", 10]]
    with pytest.raises(ValueError):
        temperature_data_from_json(data, orient="NOT_ALLOWED")


def test_temperature_data_to_csv(sample_metadata):
    series = pd.Series(10, index=pd.DatetimeIndex(["2017-01-01T00:00:00Z"]))
    with TemporaryFile("w+") as f:
        temperature_data_to_csv(series, f)
        f.seek(0)
        assert f.read() == ("dt,temperature\n" "2017-01-01,10\n")
