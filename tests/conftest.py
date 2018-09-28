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
from pkg_resources import resource_stream

import pandas as pd
import pytest

from eemeter import load_sample


@pytest.fixture
def sample_metadata():
    with resource_stream("eemeter.samples", "metadata.json") as f:
        metadata = json.loads(f.read().decode("utf-8"))
    return metadata


def _from_sample(sample):
    meter_data, temperature_data, metadata = load_sample(sample)
    return {
        "meter_data": meter_data,
        "temperature_data": temperature_data,
        "blackout_start_date": metadata["blackout_start_date"],
        "blackout_end_date": metadata["blackout_end_date"],
    }


@pytest.fixture
def il_electricity_cdd_hdd_hourly():
    return _from_sample("il-electricity-cdd-hdd-hourly")


@pytest.fixture
def il_electricity_cdd_hdd_daily():
    return _from_sample("il-electricity-cdd-hdd-daily")


@pytest.fixture
def il_electricity_cdd_hdd_billing_monthly():
    return _from_sample("il-electricity-cdd-hdd-billing_monthly")


@pytest.fixture
def il_electricity_cdd_hdd_billing_bimonthly():
    return _from_sample("il-electricity-cdd-hdd-billing_bimonthly")
