import json
from pkg_resources import resource_stream

import pandas as pd
import pytest

from eemeter import (
    meter_data_from_csv,
    temperature_data_from_csv,
)


@pytest.fixture
def sample_metadata():
    with resource_stream('eemeter.samples', 'metadata.json') as f:
        metadata = json.loads(f.read().decode('utf-8'))
    return metadata


def _from_metadata(metadata, key, freq=None):
    meter_item = metadata[key]
    meter_data_filename = meter_item['meter_data_filename']
    with resource_stream('eemeter.samples', meter_data_filename) as f:
        meter_data = meter_data_from_csv(f, gzipped=True, freq=freq)
    temperature_filename = meter_item['temperature_filename']
    with resource_stream('eemeter.samples', temperature_filename) as f:
        temperature_data = temperature_data_from_csv(f, gzipped=True)
    return {
        'meter_data': meter_data,
        'temperature_data': temperature_data,
        'blackout_start_date': pd.Timestamp(meter_item['blackout_start_date']).tz_localize('UTC'),
        'blackout_end_date': pd.Timestamp(meter_item['blackout_end_date']).tz_localize('UTC'),
    }


@pytest.fixture
def il_electricity_cdd_hdd_hourly(sample_metadata):
    return _from_metadata(
        sample_metadata, 'il-electricity-cdd-hdd-hourly', freq='hourly')


@pytest.fixture
def il_electricity_cdd_hdd_daily(sample_metadata):
    return _from_metadata(
        sample_metadata, 'il-electricity-cdd-hdd-daily', freq='daily')


@pytest.fixture
def il_electricity_cdd_hdd_billing_monthly(sample_metadata):
    return _from_metadata(
        sample_metadata, 'il-electricity-cdd-hdd-billing_monthly')


@pytest.fixture
def il_electricity_cdd_hdd_billing_bimonthly(sample_metadata):
    return _from_metadata(
        sample_metadata, 'il-electricity-cdd-hdd-billing_bimonthly')


