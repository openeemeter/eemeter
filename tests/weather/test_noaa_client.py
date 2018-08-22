from io import BytesIO
import numpy as np
import re
import pkg_resources
import tempfile

import pandas as pd
import pytest
import pytz
import json
from eeweather.testing import (
    MockKeyValueStoreProxy,
    MockTMY3RequestProxy,
    MockCZ2010RequestProxy,
)

from eemeter.weather import WeatherSource


def test_noaa_client():
    '''
    This is a station that previously provided data only up to
    May 31, 2017.
    '''
    station = '720406'
    weather_source = WeatherSource(station,
        normalized=False, use_cz2010=False)

    index = pd.date_range(
        '2017-01-01', periods=365*24, freq='H', tz=pytz.UTC)

    tempF = weather_source.indexed_temperatures(index, "degF")
    assert np.shape(tempF)==(8760,)
    assert np.shape(tempF.dropna())==(8757,)

