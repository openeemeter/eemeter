from eemeter.processors.collector import LogCollector


def test_basic_usage():

    def do_thing(logger, returnme):
        logger.debug("TEST")
        return returnme

    collector = LogCollector()
    with collector.collect_logs("key") as logger:
        result = do_thing(logger, "returnme")

    assert len(collector.items["key"]) == 1
    assert collector.items["key"][0].endswith("TEST")
    assert "DEBUG" in collector.items["key"][0]
    assert result == "returnme"
