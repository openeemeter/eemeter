import base64
import json
import os
import os.path

from io import BytesIO

import pytest
from mock import patch

import pandas as pd
from numpy.testing import assert_allclose

from eemeter.weather.clients import NOAAClient

HERE = os.path.abspath(os.path.dirname(__file__))
FIXTURE_FILE_NAME = 'noaa_responses.jsonl'
FIXTURE_PATH = os.path.join(HERE, 'fixtures', FIXTURE_FILE_NAME)


@pytest.fixture
def unplugged_noaa_client():
    """ in tests, get a NOAAClient that no longer needs FTP/Internet (by mocking one method).

    This technique is *inspired by* the wonderful VCR.py library,
    but that only works with HTTP/S. So we must do our own test harness for this.

    This relies on a previous procedure to be run, to capture and save NOAA's responses.
    That procedure is in __main__ below - so run this file as a script to get a new capture.

    The procedure uses a hook added into NOAAClient implementation, that saves out NOAA's
    responses (from FTP at time of writing), to a file in /tmp (or your OS's equivalent).
    It saves it in a JSONL format (line-separated JSON: where each line is a JSON object).

    The other thing about the formatting: the 'response_base64' field is base64-encoded.
    This is the exact response from NOAA's server, may be gzip format etc. We are testing
    NOAAClient's ability to handle that, so we save it to file in its original form,
    except base64-encoded so that saving and loading the file is predictable and stable.
    """
    captured_responses = {}

    with open(FIXTURE_PATH) as f:
        jsonl_lines = f.readlines()
        for entry_json_string in jsonl_lines:
            entry = json.loads(entry_json_string.strip())
            filename_format = entry['filename_format']
            station_ids = entry['station_ids']
            year = entry['year']
            raw_response = base64.b64decode(entry['response_base64'])
            captured_responses[(filename_format, tuple(station_ids), year)] = raw_response

    def _mock_retrieve_file(filename_format, station_ids, year):
        """ can mock the _retrieve_file() function, which is when NOAAClient talks to NOAA FTP.
        """
        bytes_io = BytesIO()
        raw_response = captured_responses[(filename_format, tuple(station_ids), year)]
        bytes_io.write(raw_response)
        return bytes_io, True

    with patch.object(NOAAClient, '_retrieve_file') as _retrieve_file:
        _retrieve_file.side_effect = _mock_retrieve_file
        client = NOAAClient()
        yield client


def test_isd_data(unplugged_noaa_client):
    client = unplugged_noaa_client
    data = client.get_isd_data('724464', '2011')
    assert data.shape == (17544,)
    ts = pd.Timestamp('2011-01-01 00:00:00+0000', tz='UTC')
    assert_allclose(data[ts], -2.0)


def test_gsod_data(unplugged_noaa_client):
    client = unplugged_noaa_client
    data = client.get_gsod_data('724464', '2011')
    assert data.shape == (365,)
    ts = pd.Timestamp('2011-01-01 00:00:00+0000', tz='UTC')
    assert_allclose(data[ts], -6.9444444444444446)


if __name__ == "__main__":
    """ You can call this file as an executable script, to record out some test cases.

    What's it for? See explanation in `unplugged_noaa_client` fixture.
    """
    import logging
    import tempfile
    import time

    from eemeter.log import setup_logging

    logger = logging.getLogger('eemeter.tests.weather.test_noaa_client.__main__')
    setup_logging(eemeter_log_level='DEBUG', allow_console_logging=True)

    noaa_client = NOAAClient()

    # TODO(hangtwenty) create a list of test cases of (station, year, expected_data_shape, ...)
    # that could be used to parametrize the test cases above... then use same list of test cases
    # here, and we can save data for all of them. then we can get even more coverage breadth!
    station = '724464'
    year = '2011'

    tmp = '/tmp' if os.path.exists('/tmp') else tempfile.gettempdir()
    CAPTURED = os.path.join(tmp, FIXTURE_FILE_NAME)

    if os.path.exists(CAPTURED):
        logger.warning('moving older {!r} ...'.format(CAPTURED))
        os.rename(CAPTURED, "{0}_old_{1}".format(CAPTURED, int(time.time())))

    noaa_client._dump_responses_to_jsonl_path = CAPTURED
    noaa_client.get_isd_data(station=station, year=year)
    noaa_client.get_gsod_data(station=station, year=year)

    logger.info("NEW FIXTURES FILE: {!r}".format(noaa_client._dump_responses_to_jsonl_path))
    logger.info("To update the test suite with this fixture, move the file into place:\n"
                "mv {!r} {!r}".format(CAPTURED, FIXTURE_PATH))

