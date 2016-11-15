import tempfile
from eemeter.weather.cache import SqlJSONStore
from datetime import datetime
import pytz


def test_basic_usage():
    tmpdir = tempfile.mkdtemp()
    url = "sqlite:///{}/weather_cache.db".format(tmpdir)
    s = SqlJSONStore(url)

    assert s.key_exists("a") is False

    data = s.retrieve_json("a")
    assert data is None

    s.save_json("a", {"b": [1, "two", 3.0]})
    assert s.key_exists("a") is True

    data = s.retrieve_json("a")
    assert len(data["b"]) == 3
    assert data["b"][0] == 1
    assert data["b"][1] == "two"
    assert data["b"][2] == 3.0

    dt1 = pytz.UTC.localize(s.retrieve_datetime("a"))
    # assertion might fail if midnight dec 31
    assert dt1.date() == datetime.utcnow().date()

    # update
    s.save_json("a", ["updated"])
    data = s.retrieve_json("a")
    assert data[0] == "updated"

    s.clear()
    assert s.key_exists("a") is False

    assert str(s) == 'SqlJSONStore("{}")'.format(url)

    s.save_json("a", "b")
    s.save_json("b", "c")
    assert s.key_exists("a") is True
    assert s.key_exists("b") is True
    s.clear("b")
    assert s.key_exists("a") is True
    assert s.key_exists("b") is False

