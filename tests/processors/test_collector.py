from eemeter.processors.collector import LogCollector


def test_basic_usage():

    def do_thing(logger, returnme):
        logger.debug("TEST")
        return returnme

    collector = LogCollector()
    with collector.collect_logs("key") as logger:
        result = do_thing(logger, "returnme")

    assert collector.items["key"].endswith("TEST\n")
    assert "DEBUG" in collector.items["key"]
    assert result == "returnme"
