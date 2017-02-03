import tempfile

import pytest
import pandas as pd
import numpy as np
import pytz

from eemeter.modeling.models.billing import BillingElasticNetCVModel
from eemeter.modeling.formatters import ModelDataBillingFormatter
from eemeter.structures import EnergyTrace
from eemeter.weather import ISDWeatherSource
from eemeter.testing.mocks import MockWeatherClient


@pytest.fixture
def trace():
    index = pd.DatetimeIndex(
        ["2012-06-06", "2012-07-06", "2012-08-06", "2012-09-06"],
        dtype='datetime64[ns, UTC]', freq=None)

    data = pd.DataFrame(
        {
            "value": [1, 1, 1, np.nan],
            "estimated": [False, False, False, False]
        }, index=index, columns=['value', 'estimated'])

    return EnergyTrace(
        interpretation="NATURAL_GAS_CONSUMPTION_SUPPLIED",
        unit="THERM", data=data)


@pytest.fixture
def mock_isd_weather_source():
    tmp_url = "sqlite:///{}/weather_cache.db".format(tempfile.mkdtemp())
    ws = ISDWeatherSource("722880", tmp_url)
    ws.client = MockWeatherClient()
    return ws


def test_basic_usage(trace, mock_isd_weather_source):
    formatter = ModelDataBillingFormatter()
    model = BillingElasticNetCVModel(65, 65)

    formatted_input_data = formatter.create_input(
        trace, mock_isd_weather_source)

    outputs = model.fit(formatted_input_data)
    assert 'upper' in outputs
    assert 'lower' in outputs
    assert 'n' in outputs
    assert 'r2' in outputs
    assert 'rmse' in outputs
    assert 'cvrmse' in outputs
    assert 'model_params' in outputs

    index = pd.date_range(
        '2011-01-01', freq='H', periods=365 * 24, tz=pytz.UTC)
    formatted_predict_data = formatter.create_demand_fixture(
        index, mock_isd_weather_source)

    outputs, variance = model.predict(formatted_predict_data, summed=False)
    assert outputs.shape == (365,)
    assert variance > 0

    index = pd.date_range('2011-01-01', freq='D', periods=365, tz=pytz.UTC)
    formatted_predict_data = formatter.create_demand_fixture(
        index, mock_isd_weather_source)

    outputs, variance = model.predict(formatted_predict_data, summed=False)
    assert outputs.shape == (365,)
    assert variance > 0

    outputs, variance = model.predict(formatted_predict_data, summed=True)
    assert outputs > 0
    assert variance > 0

    assert "ModelDataBillingFormatter" in str(ModelDataBillingFormatter)
