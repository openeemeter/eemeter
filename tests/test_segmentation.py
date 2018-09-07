import pytest

import pandas as pd

from eemeter.segmentation import segment_time_series


@pytest.fixture
def index_8760():
    return pd.date_range("2017-01-01", periods=365 * 24, freq="H", tz="UTC")


def test_segment_time_series_invalid_type(index_8760):
    with pytest.raises(ValueError):
        segment_time_series(index_8760, segment_type="unknown")


def test_segment_time_series_single(index_8760):
    weights = segment_time_series(index_8760, segment_type="single")
    assert list(weights.columns) == ["all"]
    assert weights.shape == (8760, 1)
    assert weights.sum().sum() == 8760.0


def test_segment_time_series_one_month(index_8760):
    weights = segment_time_series(index_8760, segment_type="one_month")
    assert list(weights.columns) == [
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec",
    ]
    assert weights.shape == (8760, 12)
    assert weights.sum().sum() == 8760.0


def test_segment_time_series_three_month(index_8760):
    weights = segment_time_series(index_8760, segment_type="three_month")
    assert list(weights.columns) == [
        "dec-jan-feb",
        "jan-feb-mar",
        "feb-mar-apr",
        "mar-apr-may",
        "apr-may-jun",
        "may-jun-jul",
        "jun-jul-aug",
        "jul-aug-sep",
        "aug-sep-oct",
        "sep-oct-nov",
        "oct-nov-dec",
        "nov-dec-jan",
    ]
    assert weights.shape == (8760, 12)
    assert weights.sum().sum() == 26280.0


def test_segment_time_series_three_month_weighted(index_8760):
    weights = segment_time_series(index_8760, segment_type="three_month_weighted")
    assert list(weights.columns) == [
        "dec-jan-feb-weighted",
        "jan-feb-mar-weighted",
        "feb-mar-apr-weighted",
        "mar-apr-may-weighted",
        "apr-may-jun-weighted",
        "may-jun-jul-weighted",
        "jun-jul-aug-weighted",
        "jul-aug-sep-weighted",
        "aug-sep-oct-weighted",
        "sep-oct-nov-weighted",
        "oct-nov-dec-weighted",
        "nov-dec-jan-weighted",
    ]
    assert weights.shape == (8760, 12)
    assert weights.sum().sum() == 17520.0
