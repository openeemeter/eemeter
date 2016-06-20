from eemeter.processors.collector import Collector
from eemeter.processors.collector import collects


def test_basic_usage():

    @collects()
    def do_thing(returnme):
        return returnme, {"thing_done": "abc"}

    collector = Collector()
    with collector.collect("key") as c:
        result = do_thing(c, "returnme")

    assert collector.items["key"]["thing_done"] == "abc"
    assert result == "returnme"
