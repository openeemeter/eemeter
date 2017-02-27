import tempfile
from datetime import datetime, timedelta

import pytest
import pandas as pd
import numpy as np
from numpy.testing import assert_allclose
import pytz

from eemeter.weather import ISDWeatherSource
from eemeter.testing.mocks import MockWeatherClient
from eemeter.modeling.formatters import (
    ModelDataFormatter,
    ModelDataBillingFormatter
)
from eemeter.structures import EnergyTrace
from eemeter.modeling.models import CaltrackMonthlyModel


@pytest.fixture
def mock_isd_weather_source():
    tmp_url = "sqlite:///{}/weather_cache.db".format(tempfile.mkdtemp())
    ws = ISDWeatherSource("722880", tmp_url)
    ws.client = MockWeatherClient()
    return ws


@pytest.fixture
def daily_trace():
    data = {
        "value": np.tile(1, (365,)),
        "estimated": np.tile(False, (365,)),
    }
    columns = ["value", "estimated"]
    index = pd.date_range('2000-01-01', periods=365, freq='D', tz=pytz.UTC)
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
        datetime(2011, 6, 1, tzinfo=pytz.UTC) + \
        timedelta(days=30*i) for i in range(13) ]
    df = pd.DataFrame(data, index=index, columns=columns)
    return EnergyTrace("ELECTRICITY_CONSUMPTION_SUPPLIED", df, unit="KWH")


@pytest.fixture
def input_df(mock_isd_weather_source, daily_trace):
    mdf = ModelDataFormatter("D")
    return mdf.create_input(daily_trace, mock_isd_weather_source)


@pytest.fixture
def input_billing_df(mock_isd_weather_source, billing_trace):
    mdbf = ModelDataBillingFormatter()
    return mdbf.create_input(billing_trace, mock_isd_weather_source)


def test_basic(input_df):
    m = CaltrackMonthlyModel(fit_cdd=True)
    assert str(m).startswith("Caltrack")
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

    assert m.n == 12
    assert 'formula' in m.params
    assert 'X_design_info' in m.params
    assert 'coefficients' in m.params
    assert m.r2 == 0.0
    assert_allclose(m.rmse, 0., rtol=1e-5, atol=1e-5)
    assert m.y.shape == (12, 1)

    predict, variance = m.predict(input_df, summed=False)

    assert predict.shape == (365,)
    assert_allclose(predict[datetime(2000, 1, 1, tzinfo=pytz.UTC)], 31.)
    assert all(variance > 0)

    assert m.n == 12
    assert 'formula' in m.params
    assert 'X_design_info' in m.params
    assert 'coefficients' in m.params
    assert m.r2 == 0.0
    assert_allclose(m.rmse, 0.00000000000, rtol=1e-5, atol=1e-5)
    assert m.y.shape == (12, 1)

    predict, variance = m.predict(input_df)

    assert_allclose(predict, 365.)
    assert variance > 0


def test_basic_billing(input_billing_df, mock_isd_weather_source):
    m = CaltrackMonthlyModel(fit_cdd=True)
    assert str(m).startswith("Caltrack")
    assert m.n is None
    assert m.params is None
    assert m.r2 is None
    assert m.rmse is None
    assert m.y is None

    output = m.fit(input_billing_df)

    assert "r2" in output
    assert "rmse" in output
    assert "cvrmse" in output
    assert "model_params" in output
    assert "n" in output

    assert 'formula' in m.params
    assert 'X_design_info' in m.params
    assert 'coefficients' in m.params

    index = pd.date_range('2011-01-01', freq='D', periods=365, tz=pytz.UTC)
    formatter = ModelDataBillingFormatter()
    formatted_predict_data = formatter.create_demand_fixture(
        index, mock_isd_weather_source)

    outputs, variance = m.predict(formatted_predict_data, summed=False)
    assert outputs.shape == (365,)
    assert all(variance > 0)

    outputs, variance = m.predict(formatted_predict_data, summed=True)
    assert outputs > 0
    assert variance > 0
