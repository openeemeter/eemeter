import tempfile

import pytest
from numpy.testing import assert_allclose

from eemeter.modeling.formatters import (
    ModelDataFormatter,
    ModelDataBillingFormatter,
)
from eemeter.ee.derivatives import annualized_weather_normal
from eemeter.testing.mocks import MockModel, MockWeatherClient
from eemeter.weather import TMY3WeatherSource


@pytest.fixture
def mock_tmy3_weather_source():
    tmp_dir = tempfile.mkdtemp()
    ws = TMY3WeatherSource("724838", tmp_dir, preload=False)
    ws.client = MockWeatherClient()
    ws._load_data()
    return ws


def test_daily(mock_tmy3_weather_source):
    formatter = ModelDataFormatter("D")
    model = MockModel()
    output = annualized_weather_normal(
        formatter, model, mock_tmy3_weather_source)

    assert_allclose(output['annualized_weather_normal'][:4],
                    (365, 1, 1, 365))

    serialized = output['annualized_weather_normal'][4]
    assert len(serialized) == 365
    assert serialized['2015-01-01T00:00:00+00:00'] == 32


def test_monthly(mock_tmy3_weather_source):
    formatter = ModelDataBillingFormatter()
    model = MockModel()
    output = annualized_weather_normal(
        formatter, model, mock_tmy3_weather_source)

    assert_allclose(output['annualized_weather_normal'][:4],
                    (365, 1, 1, 365))

    serialized = output['annualized_weather_normal'][4]
    assert len(serialized) == 365
    assert serialized['2015-01-01T00:00:00+00:00'] == 32
