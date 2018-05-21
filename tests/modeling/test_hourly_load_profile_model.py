import tempfile
from datetime import datetime, timedelta

import pytest
import pandas as pd
import numpy as np
from numpy.testing import assert_allclose
import pytz

from eemeter.modeling.formatters import (
    ModelDataFormatter,
    ModelDataBillingFormatter
)
from eemeter.modeling.exceptions import (
    DataSufficiencyException,
)
from eemeter.structures import EnergyTrace
from eemeter.modeling.models import HourlyLoadProfileModel
from eemeter.weather import WeatherSource


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
def mock_isd_weather_source():
    ws = WeatherSource('722880', False, False)
    return ws


@pytest.fixture
def hourly_trace():
    data = {
        "value": np.arange(365*24),
        "estimated": np.tile(False, (365*24,)),
    }
    columns = ["value", "estimated"]
    index = pd.date_range('2001-01-01', periods=365*24, freq='H', tz=pytz.UTC)
    df = pd.DataFrame(data, index=index, columns=columns)
    return EnergyTrace("ELECTRICITY_CONSUMPTION_SUPPLIED", df, unit="KWH")


@pytest.fixture
def billing_trace():
    data = {
        "value": [1, 1, 1, 1, np.nan] + [1,]*13,
        "estimated": [False, False, True, False, False] + [False,]*13
    }
    columns = ["value", "estimated"]
    index = [
        datetime(2011, 1, 1, tzinfo=pytz.UTC),
        datetime(2011, 2, 1, tzinfo=pytz.UTC),
        datetime(2011, 3, 2, tzinfo=pytz.UTC),
        datetime(2011, 4, 3, tzinfo=pytz.UTC),
        datetime(2011, 4, 29, tzinfo=pytz.UTC),
    ] + [
        datetime(2011, 6, 1, tzinfo=pytz.UTC) + timedelta(days=30*i)
        for i in range(13)
    ]
    df = pd.DataFrame(data, index=index, columns=columns)
    return EnergyTrace("ELECTRICITY_CONSUMPTION_SUPPLIED", df, unit="KWH")


@pytest.fixture
def input_df(monkeypatch_temperature_data, hourly_trace,
    mock_isd_weather_source):
    mdf = ModelDataFormatter("H")
    return mdf.create_input(hourly_trace, mock_isd_weather_source)


@pytest.fixture
def input_billing_df(monkeypatch_temperature_data, billing_trace,
        mock_isd_weather_source):
    mdbf = ModelDataBillingFormatter()
    return mdbf.create_input(billing_trace, mock_isd_weather_source)


def test_basic_billing(input_billing_df, monkeypatch_temperature_data):
    m = HourlyLoadProfileModel(fit_cdd=True)
    assert str(m).startswith("HourlyLoadProfileModel")
    assert m.n is None
    assert m.params is None
    assert m.r2 is None
    assert m.rmse is None
    assert m.y is None

    with pytest.raises(DataSufficiencyException) as e:
        m.fit(input_billing_df)
    message = str(e.value)
    assert message == (
        'Billing data is not appropriate for this model'
    )


def test_basic_hourly(input_df, monkeypatch_temperature_data,
    mock_isd_weather_source):
    m = HourlyLoadProfileModel(fit_cdd=True)
    assert str(m).startswith("HourlyLoadProfileModel")
    assert m.n is None
    assert m.params is None
    assert m.r2 is None
    assert m.rmse is None
    assert m.y is None

    output = m.fit(input_df)

    assert "r2" in output
    assert "rmse" in output
    assert "cvrmse" in output
    assert "model_params" in output
    assert "n" in output

    assert 'formula' in m.params
    assert 'X_design_info' in m.params

    index = pd.date_range('2011-01-01', freq='H', periods=365*24, tz=pytz.UTC)
    formatter = ModelDataFormatter("H")
    formatted_predict_data = formatter.create_demand_fixture(index,
        mock_isd_weather_source)

    outputs, variance = m.predict(formatted_predict_data, summed=False)
    assert outputs.shape == (365*24,)
    assert all(variance > 0)

    outputs, variance = m.predict(formatted_predict_data, summed=True)
    assert outputs > 0
    assert variance > 0
