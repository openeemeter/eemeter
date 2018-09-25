from eemeter import get_version


def test_get_version():
    assert get_version().startswith("2")
