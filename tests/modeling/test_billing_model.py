import tempfile

import pytest
import pandas as pd
import numpy as np
import pytz

from eemeter.modeling.models.billing import BillingElasticNetCVModel
from eemeter.modeling.formatters import ModelDataBillingFormatter
from eemeter.structures import EnergyTrace


def _fake_temps(usaf_id, start, end, normalized, use_cz2010):
    # sinusoidal fake temperatures in degC
    dates = pd.date_range(start, end, freq='H', tz=pytz.UTC)
    num_years = end.year - start.year + 1
    n = dates.shape[0]
    avg_temp = 15
    temp_range = 15
    period_offset = - (2 * np.pi / 3)
    temp_offsets = np.sin(
        (2 * np.pi * num_years * np.arange(n) / n) + period_offset)
    temps = avg_temp + (temp_range * temp_offsets)
    return pd.Series(temps, index=dates, dtype=float)


@pytest.fixture
def monkeypatch_temperature_data(monkeypatch):
    monkeypatch.setattr(
        'eemeter.weather.eeweather_wrapper._get_temperature_data_eeweather',
        _fake_temps
    )


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


def test_basic_usage(trace, monkeypatch_temperature_data):
    formatter = ModelDataBillingFormatter()
    model = BillingElasticNetCVModel(65, 65)

    formatted_input_data = formatter.create_input(trace)

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
    formatted_predict_data = formatter.create_demand_fixture(index)

    outputs, variance = model.predict(formatted_predict_data, summed=False)
    assert outputs.shape == (365,)
    assert variance > 0

    index = pd.date_range('2011-01-01', freq='D', periods=365, tz=pytz.UTC)
    formatted_predict_data = formatter.create_demand_fixture(index)

    outputs, variance = model.predict(formatted_predict_data, summed=False)
    assert outputs.shape == (365,)
    assert variance > 0

    outputs, variance = model.predict(formatted_predict_data, summed=True)
    assert outputs > 0
    assert variance > 0

    assert "ModelDataBillingFormatter" in str(ModelDataBillingFormatter)
