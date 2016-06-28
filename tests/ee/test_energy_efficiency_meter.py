from datetime import datetime
import tempfile

import pytz
import pytest
import pandas as pd
import numpy as np
from numpy.testing import assert_allclose

from eemeter.structures import (
    Project,
    ZIPCodeSite,
    EnergyTraceSet,
    EnergyTrace,
    Intervention,
)
from eemeter.ee.meter import EnergyEfficiencyMeter
from eemeter.testing.mocks import MockWeatherClient
from eemeter.weather import TMY3WeatherSource


@pytest.fixture
def daily_data():
    index = pd.date_range('2012-01-01', periods=365*4, freq='D', tz=pytz.UTC)
    data = {
        "value": np.tile(1, (365 * 4,)),
        "estimated": np.tile(False, (365 * 4,))
    }
    return pd.DataFrame(data, index=index, columns=["value", "estimated"])


@pytest.fixture
def energy_trace_set(daily_data):
    energy_trace_set = EnergyTraceSet([
        EnergyTrace('ELECTRICITY_CONSUMPTION_SUPPLIED', data=daily_data,
                    unit='kWh'),
    ])
    return energy_trace_set


@pytest.fixture
def interventions():
    interventions = [
        Intervention(datetime(2014, 1, 1, tzinfo=pytz.UTC),
                     datetime(2014, 2, 1, tzinfo=pytz.UTC)),
    ]
    return interventions


@pytest.fixture
def project(energy_trace_set, interventions):
    site = ZIPCodeSite("02138")
    project = Project(energy_trace_set, interventions, site)
    return project


@pytest.fixture
def mock_tmy3_weather_source():
    tmp_dir = tempfile.mkdtemp()
    ws = TMY3WeatherSource("724838", tmp_dir, preload=False)
    ws.client = MockWeatherClient()
    ws._load_data()
    return ws


def test_basic_usage(project, mock_tmy3_weather_source):
    meter = EnergyEfficiencyMeter()
    results = meter.evaluate(project,
                             weather_normal_source=mock_tmy3_weather_source)

    project_results = results['project']
    assert project_results['modeling_period_groups'] == \
        [('baseline', 'reporting')]
    assert ('baseline', '0') in project_results['modeled_trace_selectors']
    assert ('reporting', '0') in project_results['modeled_trace_selectors']
    assert project_results['trace_interpretations'] == \
        {'0': 'ELECTRICITY_CONSUMPTION_SUPPLIED'}
    assert project_results['modeling_periods'][0] == 'baseline'
    assert project_results['modeling_periods'][1] == 'reporting'
    assert_allclose(project_results[
        'total_baseline_normal_annual_electricity_consumption_kWh'],
        (382.191016124, 0.8127655376, 0.9400387554, 364))
    assert_allclose(project_results[
        'total_reporting_normal_annual_electricity_consumption_kWh'],
        (371.236190303, 0.7646731915, 0.8901040637, 334))
    assert_allclose(project_results[
        'total_baseline_normal_annual_fuel_consumption_kWh'],
        (382.191016124, 0.8127655376, 0.9400387554, 364))
    assert_allclose(project_results[
        'total_reporting_normal_annual_fuel_consumption_kWh'],
        (371.236190303, 0.7646731915, 0.8901040637, 334))

    trace_results = results['traces']
    trace1 = trace_results[('baseline', '0')]
    assert 'cvrmse' in trace1
    assert 'n' in trace1
    assert 'upper' in trace1
    assert 'annualized_weather_normal' in trace1
    assert 'lower' in trace1
    assert 'rmse' in trace1
    assert 'r2' in trace1
    assert 'model_params' in trace1
    trace2 = trace_results[('reporting', '0')]
    assert 'cvrmse' in trace2
    assert 'n' in trace2
    assert 'upper' in trace2
    assert 'annualized_weather_normal' in trace2
    assert 'lower' in trace2
    assert 'rmse' in trace2
    assert 'r2' in trace2
    assert 'model_params' in trace2

    logs = results['logs']
    assert len(logs['get_weather_source']) == 2
    assert len(logs['get_weather_normal_source']) == 1
    assert len(logs['get_modeling_period_set']) == 1
    assert len(logs['get_energy_modeling_dispatches']) == 4
    assert len(logs['handle_dispatches']) == 0
