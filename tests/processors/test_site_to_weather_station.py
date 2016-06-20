from eemeter.processors import site_to_weather_source

from eemeter.processors.collector import Collector
from eemeter.structures import ZIPCodeSite


def test_basic_usage():
    site = ZIPCodeSite("91104")

    collector = Collector()
    with collector.collect("site_to_weather_source") as c:
        weather_source = site_to_weather_source(c, site)

    assert weather_source.station == "722880"
    assert "site_to_weather_source" in collector.items

    items = collector.items["site_to_weather_source"]
    assert "zipcode_to_ISD_station" in items
    assert "ISD_station_to_weather_source" in items
