from eemeter.structures import ZIPCodeLocation
import pytest


def test_str():
    location = ZIPCodeLocation("91104")
    assert location.zipcode == "91104"

def test_int():
    location = ZIPCodeLocation(91104)
    assert location.zipcode == "91104"

def test_invalid_str():
    with pytest.raises(ValueError):
        location = ZIPCodeLocation("BLAH")

    with pytest.raises(ValueError):
        location = ZIPCodeLocation("BLAHH")

    with pytest.raises(ValueError):
        location = ZIPCodeLocation("BLAHHH")

    with pytest.raises(ValueError):
        location = ZIPCodeLocation("012345")

    with pytest.raises(ValueError):
        location = ZIPCodeLocation("01234-0123")
