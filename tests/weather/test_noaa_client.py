from mock import patch

import pandas as pd
from numpy.testing import assert_allclose


def get_noaa_client(use_mock=True):
    if use_mock:
        with patch('eemeter.weather.clients.NOAAClient') as MockNOAAClient:
            client = MockNOAAClient()
            # TODO mock _retrieve_lines to use saved data.
            return client
    else:
        from eemeter.weather.clients import NOAAClient
        client = NOAAClient()
        return client


def test_isd_data():
    client = get_noaa_client(use_mock=True)
    data = client.get_isd_data('724464', '2011')
    assert data.shape == (17544,)
    ts = pd.Timestamp('2011-01-01 00:00:00+0000', tz='UTC')
    assert_allclose(data[ts], -2.0)


def test_gsod_data():
    client = get_noaa_client(use_mock=True)
    data = client.get_gsod_data('724464', '2011')
    assert data.shape == (365,)
    ts = pd.Timestamp('2011-01-01 00:00:00+0000', tz='UTC')
    assert_allclose(data[ts], -6.9444444444444446)


if __name__ == "__main__":
    """ Call this file as an executable script, to record out some test cases.
    """
    import logging
    from eemeter.log import setup_logging
    logger = logging.getLogger('eemeter.tests.weather.test_noaa_client.__main__')
    setup_logging(eemeter_log_level='DEBUG', allow_console_logging=True)

    # TODO(hangtwenty) wrap in a forloop that can try multiple stations/years;
    # use same list of stations/years, paired with expectations, in the tests.
    station = '724464'
    year = '2011'

    OUT = '/tmp/noaa.jsonl'

    # isd
    client = get_noaa_client(use_mock=False)
    client._dump_retrieved_lines_to_jsonl_file = OUT
    client.get_isd_data(station=station, year=year)
    logger.warning("_dump_lines_to_file: {}".format(client._dump_retrieved_lines_to_jsonl_file))

    # gsod
    client = get_noaa_client(use_mock=False)
    client._dump_retrieved_lines_to_jsonl_file = OUT
    client.get_gsod_data(station=station, year=year)
    logger.warning("_dump_lines_to_file: {}".format(client._dump_retrieved_lines_to_jsonl_file))
