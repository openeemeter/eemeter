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
from datetime import datetime

from eemeter.eemeter import HourlyBaselineData, HourlyReportingData, HourlyModel
from eemeter.common.test_data import load_test_data
import numpy as np
import pandas as pd
import pytest

_TEST_METER = 110596


@pytest.fixture
def hourly_data():
    baseline, reporting = load_test_data("hourly_treatment_data")
    return baseline.loc[_TEST_METER], reporting.loc[_TEST_METER]


def test_one(hourly_data):
    baseline, reporting = hourly_data
    baseline_data = HourlyBaselineData(baseline, is_electricity_data=True)
    reporting_data = HourlyReportingData(reporting, is_electricity_data=True)
    hm = HourlyModel().fit(baseline_data)
    p1 = hm.predict(reporting_data)
    serialized = hm.to_json()
    hm2 = HourlyModel.from_json(serialized)
    p2 = hm2.predict(reporting_data)
    assert p1.equals(p2)

def test_unaligned_data(hourly_data):
    baseline, reporting = hourly_data
    reporting.index = reporting.index.shift(8, freq="H")
    baseline_data = HourlyBaselineData(baseline, is_electricity_data=True)
    reporting_data = HourlyReportingData(reporting, is_electricity_data=True)
    hm = HourlyModel().fit(baseline_data)
    hm.predict(reporting_data)


"""TEST CASES
TODO get a couple example meters with GHI, potentially some supplemental features?
    * at least one solar and one non-solar

* good, clean data with known fit/predict numbers to check for regressions
* good meter, bad temperature
    * daily frequency temp
    * too many missing values
    * tz-naive
* good temp, bad meter
    * daily/worse frequency meter
    * too many missing values
    * tz-naive
* no GHI, attempting solar
* GHI, attempting nonsolar (warning?)
* test against supplemental data logic -> should require a flag in model to fit
* all 0s in meter data -> leads to full nan
* test valid interpolations
* test with various days removed due to interpolation during fit()
    * include day where timezone shifts in either direction
* test edge case, nearly valid, but not allowed interpolations (7 consecutive hours, etc)
    * should still happen to allow model fit, but add (and test for) DQ
* test a few DQs - baseline length, etc
* unmarked net metering flag - includes warning
* all above tests using from_series in parallel, verifying that output is identical
"""
