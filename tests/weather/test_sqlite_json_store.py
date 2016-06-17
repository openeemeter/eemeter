from eemeter.weather.cache import SqliteJSONStore
import tempfile


def test_basic_usage():
    tmpdir = tempfile.mkdtemp()
    s = SqliteJSONStore(tmpdir)

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

    s.clear()
    assert s.key_exists("a") is False
