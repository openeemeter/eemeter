from eemeter.structures import ZIPCodeSite
import pytest


def test_str():
    site = ZIPCodeSite("91104")
    assert site.zipcode == "91104"

def test_int():
    site = ZIPCodeSite(91104)
    assert site.zipcode == "91104"

def test_invalid_str():
    with pytest.raises(ValueError):
        site = ZIPCodeSite("BLAH")

    with pytest.raises(ValueError):
        site = ZIPCodeSite("BLAHH")

    with pytest.raises(ValueError):
        site = ZIPCodeSite("BLAHHH")

    with pytest.raises(ValueError):
        site = ZIPCodeSite("012345")

    with pytest.raises(ValueError):
        site = ZIPCodeSite("01234-0123")
