import tempfile

import pytest
import pandas as pd
import numpy as np
import pytz

from eemeter.modeling.models.billing import BillingElasticNetCVModel
from eemeter.modeling.formatters import ModelDataBillingFormatter
from eemeter.structures import EnergyTrace


@pytest.fixture
def trace():
    index = pd.date_range('6/6/2012','6/6/2013',freq='M',
        tz=pytz.UTC)

    data = pd.DataFrame(
        {
            "value": [1,] * 12,
            "estimated": [False,] * 12
        }, index=index, columns=['value', 'estimated'])

    return EnergyTrace(
        interpretation="NATURAL_GAS_CONSUMPTION_SUPPLIED",
        unit="THERM", data=data)


def test_basic_usage(trace, monkeypatch_temperature_data,
    mock_isd_weather_source):
    formatter = ModelDataBillingFormatter()
    model = BillingElasticNetCVModel(65, 65)

    formatted_input_data = formatter.create_input(trace,
        mock_isd_weather_source)

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
    formatted_predict_data = formatter.create_demand_fixture(index,
        mock_isd_weather_source)

    outputs, variance = model.predict(formatted_predict_data, summed=False)
    assert outputs.shape == (365,)
    assert variance > 0

    index = pd.date_range('2011-01-01', freq='D', periods=365, tz=pytz.UTC)
    formatted_predict_data = formatter.create_demand_fixture(index,
        mock_isd_weather_source)

    outputs, variance = model.predict(formatted_predict_data, summed=False)
    assert outputs.shape == (365,)
    assert variance > 0

    outputs, variance = model.predict(formatted_predict_data, summed=True)
    assert outputs > 0
    assert variance > 0

    assert "ModelDataBillingFormatter" in str(ModelDataBillingFormatter)
