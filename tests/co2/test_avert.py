import pytest

from eemeter.co2.avert import AVERTSource
from eemeter.testing import MockAVERTClient


@pytest.fixture
def mock_avert_source():
    cs = AVERTSource(2016, 'UMW')
    cs.client = MockAVERTClient()
    return cs


def test_co2_by_load(mock_avert_source):
    co2_by_load = mock_avert_source.get_co2_by_load()
    assert co2_by_load.dropna().shape == co2_by_load.shape


def test_load_by_hour(mock_avert_source):
    load_by_hour = mock_avert_source.get_load_by_hour()
    assert load_by_hour.shape[0] == 366 * 24
    assert load_by_hour.dropna().shape[0] == 366 * 24
