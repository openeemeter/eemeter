import json
from pkg_resources import resource_stream

import pandas as pd
import pytest

from eemeter import (
    load_sample,
)


@pytest.fixture
def sample_metadata():
    with resource_stream('eemeter.samples', 'metadata.json') as f:
        metadata = json.loads(f.read().decode('utf-8'))
    return metadata


@pytest.fixture
def il_electricity_cdd_hdd_hourly():
    meter_data, temperature_data, metadata = \
        load_sample('il-electricity-cdd-hdd-hourly')
    return {
        'meter_data': meter_data,
        'temperature_data': temperature_data,
        'blackout_start_date': metadata['blackout_start_date'],
        'blackout_end_date': metadata['blackout_end_date'],
    }


@pytest.fixture
def il_electricity_cdd_hdd_daily():
    meter_data, temperature_data, metadata = \
        load_sample('il-electricity-cdd-hdd-daily')
    return {
        'meter_data': meter_data,
        'temperature_data': temperature_data,
        'blackout_start_date': metadata['blackout_start_date'],
        'blackout_end_date': metadata['blackout_end_date'],
    }


@pytest.fixture
def il_electricity_cdd_hdd_billing_monthly():
    meter_data, temperature_data, metadata = \
        load_sample('il-electricity-cdd-hdd-billing_monthly')
    return {
        'meter_data': meter_data,
        'temperature_data': temperature_data,
        'blackout_start_date': metadata['blackout_start_date'],
        'blackout_end_date': metadata['blackout_end_date'],
    }

@pytest.fixture
def il_electricity_cdd_hdd_billing_bimonthly():
    meter_data, temperature_data, metadata = \
        load_sample('il-electricity-cdd-hdd-billing_bimonthly')
    return {
        'meter_data': meter_data,
        'temperature_data': temperature_data,
        'blackout_start_date': metadata['blackout_start_date'],
        'blackout_end_date': metadata['blackout_end_date'],
    }
