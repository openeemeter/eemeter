import tempfile
from datetime import datetime

import pytest
import pandas as pd
import numpy as np
from numpy.testing import assert_allclose
import pytz

from eemeter.weather import ISDWeatherSource
from eemeter.testing.mocks import MockWeatherClient
from eemeter.modeling.formatters import CaltrackFormatter
from eemeter.structures import EnergyTrace
from eemeter.modeling.models import CaltrackModel


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
def input_df(mock_isd_weather_source, daily_trace):
    mdf = CaltrackFormatter()
    return mdf.create_input(daily_trace, mock_isd_weather_source)


def test_basic(input_df):
    m = CaltrackModel(fit_cdd=True)
    assert str(m).startswith("Caltrack full")
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
    assert "upper" in output
    assert "lower" in output
    assert "n" in output

    assert m.n == 12
    assert 'formula' in m.params
    assert 'X_design_info' in m.params
    assert 'coefficients' in m.params
    assert m.r2 == 0.0
    assert_allclose(m.rmse, 0., rtol=1e-5, atol=1e-5)
    assert m.y.shape == (12, 1)

    predict, lower, upper = m.predict(input_df, summed=False)

    assert predict.shape == (12,)
    assert_allclose(predict[datetime(2000, 1, 1, tzinfo=pytz.UTC)], 31.)
    assert all(lower > 0)
    assert all(upper > 0)

    assert m.n == 12
    assert 'formula' in m.params
    assert 'X_design_info' in m.params
    assert 'coefficients' in m.params
    assert m.r2 == 0.0
    assert_allclose(m.rmse, 0.00000000000, rtol=1e-5, atol=1e-5)
    assert m.y.shape == (12, 1)

    predict, lower, upper = m.predict(input_df)

    assert_allclose(predict, 365.)
    assert lower > 0
    assert upper > 0
