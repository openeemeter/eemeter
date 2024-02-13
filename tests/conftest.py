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

# Importing required modules and functions at the top.
import json
from pkg_resources import resource_stream
import pytest

# The eemeter package's load_sample function is imported for loading sample data.
from eemeter import load_sample

# A pytest fixture to load sample metadata from a JSON file.
@pytest.fixture
def sample_metadata():
    """Load and return sample metadata from a JSON resource."""
    with resource_stream("eemeter.samples", "metadata.json") as f:
        metadata = json.loads(f.read().decode("utf-8"))
    return metadata

# Utility function to load sample data with an option for temperature unit.
def _from_sample(sample, tempF=True):
    """Load meter data, temperature data, and metadata for a given sample."""
    meter_data, temperature_data, metadata = load_sample(sample, tempF=tempF)
    return {
        "meter_data": meter_data,
        "temperature_data": temperature_data,
        "blackout_start_date": metadata["blackout_start_date"],
        "blackout_end_date": metadata["blackout_end_date"],
    }

# Fixture functions are defined for each sample data set, demonstrating reuse of the utility function.
@pytest.fixture
def il_electricity_cdd_hdd_hourly():
    """Fixture for Illinois electricity data with CDD and HDD, hourly."""
    return _from_sample("il-electricity-cdd-hdd-hourly")

@pytest.fixture
def il_electricity_cdd_hdd_daily():
    """Fixture for Illinois electricity data with CDD and HDD, daily."""
    return _from_sample("il-electricity-cdd-hdd-daily")

@pytest.fixture
def il_electricity_cdd_hdd_billing_monthly():
    """Fixture for Illinois electricity billing data with CDD and HDD, monthly."""
    return _from_sample("il-electricity-cdd-hdd-billing_monthly")

@pytest.fixture
def il_electricity_cdd_hdd_billing_bimonthly():
    """Fixture for Illinois electricity billing data with CDD and HDD, bimonthly."""
    return _from_sample("il-electricity-cdd-hdd-billing_bimonthly")

@pytest.fixture
def il_gas_hdd_only_hourly():
    """Fixture for Illinois gas data with HDD only, hourly."""
    return _from_sample("il-gas-hdd-only-hourly")

@pytest.fixture
def uk_electricity_hdd_only_hourly_sample_1():
    """Fixture for UK electricity data with HDD only, hourly sample 1."""
    return _from_sample("uk-electricity-hdd-only-hourly-sample-1", tempF=False)

@pytest.fixture
def uk_electricity_hdd_only_hourly_sample_2():
    """Fixture for UK electricity data with HDD only, hourly sample 2."""
    return _from_sample("uk-electricity-hdd-only-hourly-sample-2", tempF=False)
