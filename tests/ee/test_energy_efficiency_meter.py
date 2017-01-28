import tempfile

import pandas as pd
import pytest
import pytz

from eemeter.ee.meter import EnergyEfficiencyMeter
from eemeter.testing.mocks import MockWeatherClient
from eemeter.weather import TMY3WeatherSource
from eemeter.weather import ISDWeatherSource


@pytest.fixture
def meter_input():

    record_starts = pd.date_range(
        '2012-01-01', periods=365 * 4, freq='D', tz=pytz.UTC)

    records = [
        {
            "start": dt.isoformat(),
            "value": 1.0,
            "estimated": False
        } for dt in record_starts
    ]

    meter_input = {
        "type": "SINGLE_TRACE_SIMPLE_PROJECT",
        "trace": {
            "type": "ARBITRARY_START",
            "interpretation": "NATURAL_GAS_CONSUMPTION_SUPPLIED",
            "unit": "therm",
            "records": records
        },
        "project": {
            "type": "PROJECT_WITH_SINGLE_MODELING_PERIOD_GROUP",
            "zipcode": "91104",
            "modeling_period_group": {
                "baseline_period": {
                    "start": None,
                    "end": "2014-01-01T00:00:00+00:00"
                },
                "reporting_period": {
                    "start": "2014-02-01T00:00:00+00:00",
                    "end": None
                }
            }
        }
    }
    return meter_input


@pytest.fixture
def meter_input_strange_interpretation():

    record_starts = pd.date_range(
        '2012-01-01', periods=365 * 4, freq='D', tz=pytz.UTC)

    records = [
        {
            "start": dt.isoformat(),
            "value": 1.0,
            "estimated": False
        } for dt in record_starts
    ]

    meter_input = {
        "type": "SINGLE_TRACE_SIMPLE_PROJECT",
        "trace": {
            "type": "ARBITRARY_START",
            "interpretation": "ELECTRICITY_CONSUMPTION_NET",
            "unit": "therm",
            "records": records
        },
        "project": {
            "type": "PROJECT_WITH_SINGLE_MODELING_PERIOD_GROUP",
            "zipcode": "91104",
            "modeling_period_group": {
                "baseline_period": {
                    "start": None,
                    "end": "2014-01-01T00:00:00+00:00"
                },
                "reporting_period": {
                    "start": "2014-02-01T00:00:00+00:00",
                    "end": None
                }
            }
        }
    }
    return meter_input


@pytest.fixture
def mock_isd_weather_source():
    tmp_url = "sqlite:///{}/weather_cache.db".format(tempfile.mkdtemp())
    ws = ISDWeatherSource('722880', tmp_url)
    ws.client = MockWeatherClient()
    return ws


@pytest.fixture
def mock_tmy3_weather_source():
    tmp_url = "sqlite:///{}/weather_cache.db".format(tempfile.mkdtemp())
    ws = TMY3WeatherSource('724838', tmp_url, preload=False)
    ws.client = MockWeatherClient()
    ws._load_data()
    return ws


def test_basic_usage(meter_input, mock_isd_weather_source,
                     mock_tmy3_weather_source):

    meter = EnergyEfficiencyMeter()

    results = meter.evaluate(meter_input,
                             weather_source=mock_isd_weather_source,
                             weather_normal_source=mock_tmy3_weather_source)

    assert results['status'] == 'SUCCESS'
    assert results['failure_message'] is None
    assert len(results['logs']) == 2

    assert results['eemeter_version'] is not None
    assert results['model_class'] == 'CaltrackMonthlyModel'
    assert results['model_kwargs'] is not None
    assert results['formatter_class'] == 'ModelDataFormatter'
    assert results['formatter_kwargs'] is not None

    assert results['modeled_energy_trace'] is not None

    derivatives = results['derivatives']
    assert len(derivatives) == 138
    assert derivatives[0]['modeling_period_group'] == \
        ('baseline', 'reporting')
    assert derivatives[0]['orderable'] is None
    assert derivatives[0]['unit'] is not None
    assert derivatives[0]['value'] is not None
    assert derivatives[0]['variance'] is not None
    assert derivatives[0]['serialized_demand_fixture'] is not None

    source_series = set([(d['source'], d['series']) for d in derivatives])
    assert source_series == set([
        ('baseline_model', 'annualized_weather_normal'),
        ('baseline_model', 'annualized_weather_normal_monthly'),
        ('baseline_model', 'reporting_cumulative'),
        ('baseline_model', 'reporting_monthly'),
        ('baseline_model_minus_observed', 'reporting_cumulative'),
        ('baseline_model_minus_observed', 'reporting_monthly'),
        ('baseline_model_minus_reporting_model', 'annualized_weather_normal'),
        ('baseline_model_minus_reporting_model',
            'annualized_weather_normal_monthly'),
        ('observed', 'baseline_monthly'),
        ('observed', 'project_monthly'),
        ('observed', 'reporting_cumulative'),
        ('observed', 'reporting_monthly'),
        ('reporting_model', 'annualized_weather_normal'),
        ('reporting_model', 'annualized_weather_normal_monthly')
    ])


def test_bad_meter_input(mock_isd_weather_source, mock_tmy3_weather_source):

    meter = EnergyEfficiencyMeter()

    results = meter.evaluate({},
                             weather_source=mock_isd_weather_source,
                             weather_normal_source=mock_tmy3_weather_source)

    assert results['status'] == 'FAILURE'
    assert results['failure_message'].startswith("Meter input")


def test_strange_interpretation(meter_input_strange_interpretation,
                                mock_isd_weather_source,
                                mock_tmy3_weather_source):

    meter = EnergyEfficiencyMeter()

    results = meter.evaluate(meter_input_strange_interpretation,
                             weather_source=mock_isd_weather_source,
                             weather_normal_source=mock_tmy3_weather_source)

    assert results['status'] == 'FAILURE'
    assert results['failure_message'].startswith("Default formatter")
