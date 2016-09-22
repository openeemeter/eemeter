import tempfile
from datetime import datetime

import pytest
import pytz
from numpy.testing import assert_allclose

from eemeter.modeling.formatters import ModelDataFormatter
from eemeter.ee.derivatives import gross_predicted
from eemeter.testing.mocks import MockModel, MockWeatherClient
from eemeter.weather import ISDWeatherSource
from eemeter.structures import ModelingPeriod


@pytest.fixture
def mock_isd_weather_source():
    tmp_dir = tempfile.mkdtemp()
    ws = ISDWeatherSource("724838", tmp_dir)
    ws.client = MockWeatherClient()
    return ws


@pytest.fixture
def reporting_period_end_date():
    return ModelingPeriod("REPORTING",
                          start_date=datetime(2015, 1, 1, tzinfo=pytz.UTC),
                          end_date=datetime(2016, 6, 30, tzinfo=pytz.UTC))


@pytest.fixture
def reporting_period_no_end_date():
    return ModelingPeriod("REPORTING",
                          start_date=datetime(2015, 1, 1, tzinfo=pytz.UTC))


def test_basic_usage(mock_isd_weather_source,
                     reporting_period_end_date,
                     reporting_period_no_end_date):
    formatter = ModelDataFormatter("D")
    model = MockModel()

    output_end_date = gross_predicted(
        formatter, model, mock_isd_weather_source, reporting_period_end_date)

    output_no_end_date = gross_predicted(
        formatter, model, mock_isd_weather_source,
        reporting_period_no_end_date)

    assert_allclose(output_end_date['gross_predicted'][:4],
                    (547, 23.388031127053, 23.388031127053, 547))
    assert (
        output_end_date['gross_predicted'][0] <
        output_no_end_date['gross_predicted'][0]
    )

    serialized = output_end_date['gross_predicted'][4]
    assert len(serialized) == 547
    assert serialized['2015-01-01T00:00:00+00:00'] == 32.0
