from eemeter.weather import WeatherSourceBase
from eemeter.weather import GSODWeatherSource
from eemeter.weather import ISDWeatherSource
from eemeter.weather import TMY3WeatherSource
from eemeter.weather import CZ2010WeatherSource
from eemeter.weather import WeatherUndergroundWeatherSource

from eemeter.consumption import ConsumptionData
from eemeter.evaluation import Period

from datetime import datetime
from datetime import timedelta
import pytest
import os
import warnings

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

def test_weather_source_base(periods):
    weather_source = WeatherSourceBase()
    with pytest.raises(NotImplementedError):
        avg_temps = weather_source.average_temperature(periods, "degF")
    with pytest.raises(NotImplementedError):
        hdds = weather_source.hdd(periods, "degF", base=65)

@pytest.mark.slow
@pytest.mark.internet
def test_gsod_weather_source(periods, gsod_weather_source):
    gsod_weather_source = GSODWeatherSource(*gsod_weather_source)

    avg_temps = gsod_weather_source.average_temperature(periods,"degF")
    assert_allclose(avg_temps, [66.3833,67.803,74.445], rtol=RTOL,atol=ATOL)

    hdds = gsod_weather_source.hdd(periods,"degF",65)
    assert_allclose(hdds, [0.7,20.4,0.0], rtol=RTOL,atol=ATOL)

    cdds = gsod_weather_source.cdd(periods,"degF",65)
    assert_allclose(cdds, [42.2,107.3,292.8], rtol=RTOL,atol=ATOL)

    hdds_per_day = gsod_weather_source.hdd(periods,"degF",65,per_day=True)
    assert_allclose(hdds_per_day, [0.023,0.658,0.0], rtol=RTOL,atol=ATOL)

    cdds_per_day = gsod_weather_source.cdd(periods,"degF",65,per_day=True)
    assert_allclose(cdds_per_day, [1.406,3.461,9.445], rtol=RTOL,atol=ATOL)

@pytest.mark.slow
@pytest.mark.internet
def test_weather_underground_weather_source(periods):
    wunderground_api_key = os.environ.get('WEATHERUNDERGROUND_API_KEY')
    if wunderground_api_key:
        wu_weather_source = WeatherUndergroundWeatherSource('60605',
                                                            datetime(2012,6,1),
                                                            datetime(2012,10,1),
                                                            wunderground_api_key)

        avg_temps = wu_weather_source.average_temperature(periods,"degF")
        assert_allclose(avg_temps, [74.433,82.677,75.451], rtol=RTOL,atol=ATOL)

        hdds = wu_weather_source.hdd(periods,"degF",65)
        assert_allclose(hdds, [14.0,0.0,0.0], rtol=RTOL,atol=ATOL)

        cdds = wu_weather_source.cdd(periods,"degF",65)
        assert_allclose(cdds, [297.0,548.0,324.0], rtol=RTOL,atol=ATOL)
    else:
        warnings.warn("Skipping WeatherUndergroundWeatherSource tests. "
            "Please set the environment variable "
            "WEATHERUNDERGOUND_API_KEY to run the tests.")

@pytest.mark.slow
@pytest.mark.internet
def test_isd_weather_source(periods, isd_weather_source):
    isd_weather_source = ISDWeatherSource(*isd_weather_source)

    avg_temps = isd_weather_source.average_temperature(periods,"degF")
    assert_allclose(avg_temps, [66.576,68.047,74.697], rtol=RTOL,atol=ATOL)

    hdds = isd_weather_source.hdd(periods,"degF",65)
    assert_allclose(hdds, [0.945,24.517,0.000], rtol=RTOL,atol=ATOL)

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
@pytest.mark.internet
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

@pytest.mark.slow
@pytest.mark.internet
def test_cz2010_weather_source(periods):
    cz2010_file = os.environ.get('EEMETER_PATH_TO_ALTURAS_725958_CZ2010_CSV')
    if cz2010_file:
        cz2010_weather_source = CZ2010WeatherSource(cz2010_file)

        avg_temps = cz2010_weather_source.average_temperature(periods,"degF")
        assert_allclose(avg_temps, [68.1822,73.05548,74.315], rtol=RTOL,atol=ATOL)

        hdds = cz2010_weather_source.hdd(periods,"degF",65)
        assert_allclose(hdds, [106.3725,0.,11.775 ], rtol=RTOL,atol=ATOL)

        cdds = cz2010_weather_source.cdd(periods,"degF",65)
        assert_allclose(cdds, [51.7875,227.49,116.415], rtol=RTOL,atol=ATOL)
    else:
        warnings.warn("Skipping CZ2010WeatherSource tests. "
            "Please set the environment variable "
            "EEMETER_PATH_TO_ALTURAS_725958_CZ2010_CSV to run the tests.")

def test_generic_daily_weather_source_hdd_nan_handling():
    class GenericDailyWeatherSource(WeatherSourceBase):
        def __init__(self):
            data = {}
            offset = 4.2
            temps = 55 + 30*np.sin(np.linspace(offset,offset+2*np.pi,365)) # degF, year
            for t,dt in zip(temps,[datetime(2014,1,1) + timedelta(days=days) for days in range(365)]):
                data[dt.strftime("%Y%m%d")] = t
            self.data = data
            self._internal_unit = "degF"

        def internal_unit_datetime_average_temperature(self,dt):
            return self.data.get(dt.strftime("%Y%m%d"),np.nan)

    ws = GenericDailyWeatherSource()
    period = Period(datetime(2014,1,1),datetime(2015,1,1))

    # verify that it works to begin with.
    assert len(ws.data) == 365
    assert_allclose(ws.hdd(period,'degF',65),5527.026,rtol=RTOL,atol=ATOL)
    assert_allclose(ws.hdd(period,'degF',70),6691.383,rtol=RTOL,atol=ATOL)
    assert_allclose(ws.cdd(period,'degF',65),1850.879,rtol=RTOL,atol=ATOL)
    assert_allclose(ws.cdd(period,'degF',70),1190.235,rtol=RTOL,atol=ATOL)

    # now remove a random date, should be able to handle the nan.
    del(ws.data["20140201"])
    assert len(ws.data) == 364
    assert_allclose(ws.hdd(period,'degF',65),5487.034,rtol=RTOL,atol=ATOL)
    assert_allclose(ws.cdd(period,'degF',70),1190.235,rtol=RTOL,atol=ATOL)

