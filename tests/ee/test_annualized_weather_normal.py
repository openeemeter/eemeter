import tempfile

import pytest
from numpy.testing import assert_allclose

from eemeter.modeling.formatters import ModelDataFormatter
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


def test_basic_usage(mock_tmy3_weather_source):
    formatter = ModelDataFormatter("D")
    model = MockModel()
    output = annualized_weather_normal(
        formatter, model, mock_tmy3_weather_source)

    assert_allclose(output['annualized_weather_normal'][:4],
                    (365, 19.1049731745428, 19.1049731745428, 365))
