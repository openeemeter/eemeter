import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import pytz

from eemeter.structures import (
    ZIPCodeSite,
    EnergyTrace,
    ModelingPeriod,
    ModelingPeriodSet,
)
from eemeter.modeling.split import SplitModeledEnergyTrace
from eemeter.ee.meter import EnergyEfficiencyMeterTraceCentric
from eemeter.testing.mocks import MockWeatherClient
from eemeter.weather import TMY3WeatherSource
from eemeter.weather import ISDWeatherSource


@pytest.fixture
def daily_data():
    index = pd.date_range('2012-01-01', periods=365*4, freq='D', tz=pytz.UTC)
    data = {
        "value": np.tile(1, (365 * 4,)),
        "estimated": np.tile(False, (365 * 4,))
    }
    return pd.DataFrame(data, index=index, columns=["value", "estimated"])


@pytest.fixture
def energy_trace(daily_data):
    return EnergyTrace('ELECTRICITY_CONSUMPTION_SUPPLIED',
                       data=daily_data, unit='kWh')


@pytest.fixture
def mock_isd_weather_source():
    tmp_dir = tempfile.mkdtemp()
    ws = ISDWeatherSource("722880", tmp_dir)
    ws.client = MockWeatherClient()
    return ws


@pytest.fixture
def mock_tmy3_weather_source():
    tmp_dir = tempfile.mkdtemp()
    ws = TMY3WeatherSource("724838", tmp_dir, preload=False)
    ws.client = MockWeatherClient()
    ws._load_data()
    return ws


@pytest.fixture
def site():
    return ZIPCodeSite("02138")


@pytest.fixture
def modeling_period_set():
    modeling_period_1 = ModelingPeriod(
        "BASELINE",
        end_date=datetime(2014, 1, 1, tzinfo=pytz.UTC),
    )
    modeling_period_2 = ModelingPeriod(
        "REPORTING",
        start_date=datetime(2014, 1, 1, tzinfo=pytz.UTC),
    )
    modeling_periods = {
        "modeling_period_1": modeling_period_1,
        "modeling_period_2": modeling_period_2,
    }

    grouping = [
        ("modeling_period_1", "modeling_period_2"),
    ]

    return ModelingPeriodSet(modeling_periods, grouping)


def test_basic_usage(energy_trace, site, modeling_period_set,
                     mock_isd_weather_source, mock_tmy3_weather_source):

    meter = EnergyEfficiencyMeterTraceCentric()

    results = meter.evaluate(energy_trace, site, modeling_period_set,
                             weather_source=mock_isd_weather_source,
                             weather_normal_source=mock_tmy3_weather_source)

    assert results["status"] == "SUCCESS"
    assert results["failure_message"] is None
    assert len(results["logs"]) == 2

    assert results["eemeter_version"] == '0.4.9'
    assert results["model_class"] == 'SeasonalElasticNetCVModel'
    assert results["model_kwargs"] is not None
    assert results["formatter_class"] == 'ModelDataFormatter'
    assert results["formatter_kwargs"] is not None

    assert results["modeled_energy_trace"] is not None
    assert len(results["derivatives"]) == 2

    assert results["weather_source_station"] == '722880'
    assert results["weather_normal_source_station"] == '724838'
