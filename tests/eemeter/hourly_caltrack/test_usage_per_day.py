#!/usr/bin/env python
# -*- coding: utf-8 -*-

#  Copyright 2014-2025 OpenDSM contributors
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import pandas as pd
import pytest

from opendsm.eemeter.models.hourly_caltrack.usage_per_day import caltrack_sufficiency_criteria



@pytest.fixture
def full_year_data_quality():
    """366 daily rows of complete meter and temperature data, so the data alone spans the
    365 days the criteria expect."""
    index = pd.date_range("2019-01-01", periods=366, freq="D", tz="UTC")
    data_quality = pd.DataFrame(
        {"meter_value": 1.0, "temperature_not_null": 24.0, "temperature_null": 0.0},
        index=index,
    )

    return data_quality


def _warning_names(sufficiency):
    return [warning.qualified_name for warning in sufficiency.warnings]


def test_complete_year_passes_without_warnings(full_year_data_quality):
    sufficiency = caltrack_sufficiency_criteria(
        full_year_data_quality,
        requested_start=full_year_data_quality.index[0],
        requested_end=full_year_data_quality.index[-1],
    )

    assert sufficiency.status == "PASS"
    assert _warning_names(sufficiency) == []


def test_data_after_the_requested_end_is_a_critical_warning(full_year_data_quality):
    sufficiency = caltrack_sufficiency_criteria(
        full_year_data_quality,
        requested_start=full_year_data_quality.index[0],
        requested_end=full_year_data_quality.index[-1] - pd.Timedelta(days=10),
    )

    assert sufficiency.status == "FAIL"
    assert "eemeter.caltrack_sufficiency_criteria.extra_data_after_requested_end_date" in (
        _warning_names(sufficiency)
    )


def test_data_before_the_requested_start_is_a_critical_warning(full_year_data_quality):
    sufficiency = caltrack_sufficiency_criteria(
        full_year_data_quality,
        requested_start=full_year_data_quality.index[0] + pd.Timedelta(days=10),
        requested_end=full_year_data_quality.index[-1],
    )

    assert sufficiency.status == "FAIL"
    assert "eemeter.caltrack_sufficiency_criteria.extra_data_before_requested_start_date" in (
        _warning_names(sufficiency)
    )


def test_a_gap_before_the_requested_end_counts_toward_the_total_days(full_year_data_quality):
    """Ten requested days with no data stretch the span to 375 days, which fails the
    365-day count."""
    sufficiency = caltrack_sufficiency_criteria(
        full_year_data_quality,
        requested_start=full_year_data_quality.index[0],
        requested_end=full_year_data_quality.index[-1] + pd.Timedelta(days=10),
    )

    assert sufficiency.status == "FAIL"
    assert "eemeter.caltrack_sufficiency_criteria.incorrect_number_of_total_days" in (
        _warning_names(sufficiency)
    )
