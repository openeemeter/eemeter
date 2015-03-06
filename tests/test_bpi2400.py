from eemeter.meter import BPI2400Meter
from fixtures.consumption import bpi_2400_1
from fixtures.weather import tmy3_722880_weather_source
from fixtures.weather import gsod_722880_2012_2014_weather_source

def test_bpi2400(bpi_2400_1,
                 gsod_722880_2012_2014_weather_source,
                 tmy3_722880_weather_source):

    meter = BPI2400Meter()
    ch, elec_params, gas_params = bpi_2400_1
    result = meter.evaluate(consumption_history=ch,
                            weather_source=gsod_722880_2012_2014_weather_source,
                            weather_normal_source=tmy3_722880_weather_source)

    for c in result["consumption_history_no_estimated"].iteritems():
        assert not c.estimated

