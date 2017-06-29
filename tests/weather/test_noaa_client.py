from eemeter.weather.clients import NOAAClient
import pandas as pd
from numpy.testing import assert_allclose


def test_isod_data():
    client = NOAAClient()
    data = client.get_isd_data('724464', '2011')
    assert data.shape == (17544,)
    ts = pd.Timestamp('2011-01-01 00:00:00+0000', tz='UTC')
    assert_allclose(data[ts], -2.0)


def test_gsod_data():
    client = NOAAClient()
    data = client.get_gsod_data('724464', '2011')
    assert data.shape == (365,)
    ts = pd.Timestamp('2011-01-01 00:00:00+0000', tz='UTC')
    assert_allclose(data[ts], -6.9444444444444446)
