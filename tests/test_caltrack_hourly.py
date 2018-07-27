import json

import numpy as np
import pandas as pd
import pytest

from eemeter import (
    load_sample,
)

# E2E Test
@pytest.fixture
def utc_index():
    return pd.date_range('2011-01-01', freq='H', periods=365*24 + 1, tz='UTC')

@pytest.fixture
def temperature_data(utc_index):
    meter_data, temperature_data, metadata = \
        load_sample('il-electricity-cdd-hdd-hourly')
    return temperature_data

@pytest.fixture
def meter_data():
    meter_data, temperature_data, metadata = \
        load_sample('il-electricity-cdd-hdd-hourly')
    return meter_data

## Merge meter and temperature data
def test_merge_temperature(
        meter_data, temperature_data):
    assert False
    
## Calculate occupancy lookup table
def test_estimate_occupancy(
        meter_data, temperature_data):
    assert False
    
## Validate temperature bin endpoints and determine temperature bins
def test_assign_temperature_bins(
        meter_data, temperature_data):
    assert False
    
## Generate design matrix for weighted 3-month baseline
def test_design_matrix(
        meter_data, temperature_data):
    assert False
    
## Fit consumption model
def test_caltrack_hourly_fit(
        meter_data, temperature_data):
    assert False
    
## Use fitted model to predict counterfactual in reporting period
def test_caltrack_hourly_predict(
        meter_data, temperature_data):
    assert False
    

# Unit tests