from eemeter.weather import WeatherSourceBase
from eemeter.weather import GSODWeatherSource
from eemeter.weather import ISDWeatherSource
from eemeter.weather import TMY3WeatherSource

from eemeter.consumption import ConsumptionData
from eemeter.evaluation import Period

from pkg_resources import resource_stream
from datetime import datetime
from datetime import timedelta
import pytest
import os
import warnings
import tempfile

import numpy as np

from numpy.testing import assert_allclose

RTOL = 1e-1
ATOL = 1e-1

##### Fixtures #####

@pytest.fixture
def consumption_data_one_summer_electricity():
    records = [{"value": 0, "start": datetime(2012,6,1), "end": datetime(2012,7,1)},
               {"value": 0, "start": datetime(2012,7,1), "end": datetime(2012,8,1)},
               {"value": 0, "start": datetime(2012,8,1), "end": datetime(2012,9,1)}]
    return ConsumptionData(records,"electricity", "kWh", record_type="arbitrary")

@pytest.fixture
def periods(consumption_data_one_summer_electricity):
    return consumption_data_one_summer_electricity.periods()

@pytest.fixture(params=[(41.8955360374983,-87.6217660821178,"725340"),
                        (34.1678563835543,-118.126220490392,"722880"),
                        (42.3769095103979,-71.1247640734676,"725090"),
                        (42.3594006437094,-87.8581578622419,"725347")])
def lat_long_station(request):
    return request.param

@pytest.fixture(params=[(41.8955360374983,-87.6217660821178,"60611"),
                        (34.1678563835543,-118.126220490392,"91104"),
                        (42.3769095103979,-71.1247640734676,"02138"),
                        (42.3594006437094,-87.8581578622419,"60085"),
                        (None,None,"00000")])
def lat_long_zipcode(request):
    return request.param

@pytest.fixture(params=[('722874-93134',2012,2012),
                        ('722874',2012,2012)])
def gsod_weather_source(request):
    return request.param

@pytest.fixture(params=[('722874-93134',2012,2012),
                        ('722874',2012,2012)])
def isd_weather_source(request):
    return request.param

@pytest.fixture(params=[TMY3WeatherSource('722880')])
def tmy3_weather_source(request):
    return request.param

@pytest.fixture(params=[("60611","725340"),
                        ("91104","722880"),
                        ("02138","725090"),
                        ("60085","725347")])
def zipcode_to_station(request):
    return request.param


##### Tests #####

@pytest.mark.slow
def test_gsod_weather_source(periods, gsod_weather_source):
    gsod_weather_source = GSODWeatherSource(*gsod_weather_source)

    avg_temps = gsod_weather_source.average_temperature(periods,"degF")
    assert_allclose(avg_temps, [66.3833,67.803,74.445], rtol=RTOL,atol=ATOL)

    hdds = gsod_weather_source.hdd(periods,"degF",65)
    assert_allclose(hdds, [0.7,17.,0.0], rtol=RTOL,atol=ATOL)

    cdds = gsod_weather_source.cdd(periods,"degF",65)
    assert_allclose(cdds, [42.2,107.3,292.8], rtol=RTOL,atol=ATOL)

    hdds_per_day = gsod_weather_source.hdd(periods,"degF",65,per_day=True)
    assert_allclose(hdds_per_day, [0.023,0.658,0.0], rtol=RTOL,atol=ATOL)

    cdds_per_day = gsod_weather_source.cdd(periods,"degF",65,per_day=True)
    assert_allclose(cdds_per_day, [1.406,3.461,9.445], rtol=RTOL,atol=ATOL)

    json_data = gsod_weather_source.json()
    assert "station" in json_data
    assert type(json_data["records"][0]["datetime"]) == str

@pytest.mark.slow
def test_isd_weather_source(periods, isd_weather_source):
    isd_weather_source = ISDWeatherSource(*isd_weather_source)

    avg_temps = isd_weather_source.average_temperature(periods,"degF")
    assert_allclose(avg_temps, [66.576,68.047,74.697], rtol=RTOL,atol=ATOL)

    hdds = isd_weather_source.hdd(periods,"degF",65)
    assert_allclose(hdds, [0.61,17.1,0.000], rtol=RTOL,atol=ATOL)

    cdds = isd_weather_source.cdd(periods,"degF",65)
    assert_allclose(cdds, [42.06,107.0925,292.46837], rtol=RTOL,atol=ATOL)

    hourly_temps = isd_weather_source.hourly_temperatures(periods,"degF")
    assert_allclose(hourly_temps[0][:5],[69.98,66.92,64.04,62.96,62.96],rtol=RTOL,atol=ATOL)

    hourly_temps = isd_weather_source.hourly_temperatures(periods[0],"degF")
    assert_allclose(hourly_temps[:5],[69.98,66.92,64.04,62.96,62.96],rtol=RTOL,atol=ATOL)

    # test single period case (is iterable type error caught?)
    daily_temps = isd_weather_source.daily_temperatures(periods[0],"degF")
    assert_allclose(daily_temps[:3], [66.466,66.098,66.685], rtol=RTOL, atol=ATOL)

    # test single period case (is iterable type error caught?)
    daily_temps = isd_weather_source.daily_temperatures(periods[0],"degF")
    avg_temp = isd_weather_source.average_temperature(periods[0],"degF")
    assert_allclose(avg_temp, 66.576, rtol=RTOL, atol=ATOL)

@pytest.mark.slow
def test_tmy3_weather_source(periods, tmy3_weather_source):

    avg_temps = tmy3_weather_source.average_temperature(periods,"degF")
    assert_allclose(avg_temps, [68.1822,73.05548,74.315], rtol=RTOL,atol=ATOL)

    hdds = tmy3_weather_source.hdd(periods,"degF",65)
    assert_allclose(hdds, [10.072,0.0749,0.0], rtol=RTOL,atol=ATOL)

    cdds = tmy3_weather_source.cdd(periods,"degF",65)
    assert_allclose(cdds, [105.540,249.795,288.780], rtol=RTOL,atol=ATOL)

    hdds = tmy3_weather_source.hdd(periods,"degC",18.33)
    assert_allclose(hdds, [5.95,0.0416,0.0], rtol=RTOL,atol=ATOL)

    cdds = tmy3_weather_source.cdd(periods,"degC",18.33)
    assert_allclose(cdds, [58.63,138.775,160.433], rtol=RTOL,atol=ATOL)

def test_tmy3_weather_source_null():
    ws = TMY3WeatherSource("INVALID")
    assert_allclose(ws.tempC.values, np.zeros((365*24,)) * np.nan)

@pytest.mark.slow
def test_cache():
    cache_dir = tempfile.mkdtemp()
    ws = GSODWeatherSource('722880', cache_directory=cache_dir)

    assert "GSOD" in ws.cache_filename
    assert ".json" in ws.cache_filename

    assert ws.tempC.shape == (0,)

    assert ws.tempC.shape == (0,)

    ws.add_year(2015)
    assert ws.tempC.shape == (365,)

    ws.add_year(2013)
    assert ws.tempC.shape == (365*3,)

    ws.save_to_cache()

    # new instance, loaded from full cache
    ws = GSODWeatherSource('722880', cache_directory=cache_dir)
    assert ws.tempC.shape == (365*3,)

    # corrupt the cache
    with open(ws.cache_filename, 'w') as f:
        f.seek(1000)
        f.write("0#2]]]],,,sd,f,\\sf\\f\s34")

    # shouldn't fail - should just clear the cache
    ws = GSODWeatherSource('722880', cache_directory=cache_dir)
    assert ws.tempC.shape == (0,)

    # new instance, loaded from empty cache
    ws = GSODWeatherSource('722880', cache_directory=cache_dir)

    assert ws.tempC.shape == (0,)

    # cache still empty
    ws.load_from_cache()

    assert ws.tempC.shape == (0,)

    # write an all-null cache file
    with open(ws.cache_filename, 'w') as f:
        f.write('[["20110101", null]]')

    # new instance, loaded from empty cache
    ws = GSODWeatherSource('722880', cache_directory=cache_dir)

    assert ws.tempC.shape == (1,)
