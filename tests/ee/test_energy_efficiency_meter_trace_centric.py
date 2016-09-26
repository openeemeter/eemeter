import tempfile

import pandas as pd
import pytest
import pytz

from eemeter.ee.meter import EnergyEfficiencyMeterTraceCentric
from eemeter.testing.mocks import MockWeatherClient
from eemeter.weather import TMY3WeatherSource
from eemeter.weather import ISDWeatherSource


@pytest.fixture
def meter_input():

    record_starts = pd.date_range('2012-01-01', periods=365*4, freq='D',
                                  tz=pytz.UTC)

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
def mock_isd_weather_source():
    tmp_dir = tempfile.mkdtemp()
    ws = ISDWeatherSource('722880', tmp_dir)
    ws.client = MockWeatherClient()
    return ws


@pytest.fixture
def mock_tmy3_weather_source():
    tmp_dir = tempfile.mkdtemp()
    ws = TMY3WeatherSource('724838', tmp_dir, preload=False)
    ws.client = MockWeatherClient()
    ws._load_data()
    return ws


def test_basic_usage(meter_input, mock_isd_weather_source,
                     mock_tmy3_weather_source):

    meter = EnergyEfficiencyMeterTraceCentric()

    results = meter.evaluate(meter_input,
                             weather_source=mock_isd_weather_source,
                             weather_normal_source=mock_tmy3_weather_source)

    assert results['status'] == 'SUCCESS'
    assert results['failure_message'] is None
    assert len(results['logs']) == 2

    assert results['eemeter_version'] is not None
    assert results['model_class'] == 'SeasonalElasticNetCVModel'
    assert results['model_kwargs'] is not None
    assert results['formatter_class'] == 'ModelDataFormatter'
    assert results['formatter_kwargs'] is not None

    assert results['modeled_energy_trace'] is not None

    assert len(results['derivatives']) == 2
    assert results['derivatives'][0]["derivative_interpretation"] == \
        'annualized_weather_normal'
    assert results['derivatives'][0]["trace_interpretation"] == \
        'NATURAL_GAS_CONSUMPTION_SUPPLIED'
    assert results['derivatives'][0]["unit"] == 'THERM'
    assert results['derivatives'][0]["baseline"]["label"] == 'baseline'
    assert results['derivatives'][0]["reporting"]["label"] == 'reporting'
    assert results['derivatives'][0]["baseline"]["value"] > 0
    assert results['derivatives'][0]["reporting"]["value"] > 0
    assert results['derivatives'][1]["derivative_interpretation"] == \
        'gross_predicted'
    assert results['derivatives'][1]["baseline"]["label"] == 'baseline'
    assert results['derivatives'][1]["reporting"]["label"] == "reporting"
    assert results['derivatives'][1]["baseline"]["value"] > 0
    assert results['derivatives'][1]["reporting"]["value"] > 0

    assert results['weather_source_station'] == '722880'
    assert results['weather_normal_source_station'] == '724838'


def test_bad_meter_input(mock_isd_weather_source, mock_tmy3_weather_source):

    meter = EnergyEfficiencyMeterTraceCentric()

    results = meter.evaluate({},
                             weather_source=mock_isd_weather_source,
                             weather_normal_source=mock_tmy3_weather_source)

    assert results['status'] == 'FAILURE'
    assert results['failure_message'].startswith("Meter input")
