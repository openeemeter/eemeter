from eemeter.processors.collector import Collector
from eemeter.processors.collector import collects


def test_basic_usage():

    @collects(["thing_done"])
    def do_thing(returnme):
        return returnme, ("abc",)

    collector = Collector()
    with collector.collect("key") as c:
        result = do_thing(c, "returnme")

    assert collector.items["key"]["thing_done"] == "abc"
    assert result == "returnme"
