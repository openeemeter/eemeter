import pytest

import pandas as pd

from eemeter.segmentation import segment_time_series, iterate_segmented_dataset


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


@pytest.fixture
def dataset():
    index = pd.date_range("2017-01-01", periods=1000, freq="H", tz="UTC")
    return pd.DataFrame({"a": 1, "b": 2}, index=index, columns=["a", "b"])


def test_iterate_segmented_dataset_no_segmentation(dataset):
    iterator = iterate_segmented_dataset(dataset, segmentation=None)
    segment_name, data = next(iterator)
    assert segment_name is None
    assert list(data.columns) == ["a", "b", "weight"]
    assert data.shape == (1000, 3)
    assert data.sum().sum() == 4000

    with pytest.raises(StopIteration):
        next(iterator)


@pytest.fixture
def segmentation(dataset):
    return segment_time_series(dataset.index, segment_type="one_month")


def test_iterate_segmented_dataset_with_segmentation(dataset, segmentation):
    iterator = iterate_segmented_dataset(dataset, segmentation=segmentation)
    segment_name, data = next(iterator)
    assert segment_name == "jan"
    assert list(data.columns) == ["a", "b", "weight"]
    assert data.shape == (744, 3)
    assert data.sum().sum() == 2976.0

    segment_name, data = next(iterator)
    assert segment_name == "feb"
    assert list(data.columns) == ["a", "b", "weight"]
    assert data.shape == (256, 3)
    assert data.sum().sum() == 1024.0

    segment_name, data = next(iterator)
    assert segment_name == "mar"
    assert list(data.columns) == ["a", "b", "weight"]
    assert data.shape == (0, 3)
    assert data.sum().sum() == 0.0


def test_iterate_segmented_dataset_with_processor(dataset, segmentation):
    feature_processor_segment_names = []

    def feature_processor(
        segment_name, dataset, column_mapping=None
    ):  # rename some columns
        feature_processor_segment_names.append(segment_name)
        return dataset.rename(columns=column_mapping)

    iterator = iterate_segmented_dataset(
        dataset,
        segmentation=segmentation,
        feature_processor=feature_processor,
        feature_processor_kwargs={"column_mapping": {"a": "c", "b": "d"}},
        feature_processor_segment_name_mapping={"jan": "jan2", "feb": "feb2"},
    )
    segment_name, data = next(iterator)
    assert feature_processor_segment_names == ["jan2"]
    assert segment_name == "jan"
    assert list(data.columns) == ["c", "d", "weight"]
    assert data.shape == (744, 3)
    assert data.sum().sum() == 2976.0

    segment_name, data = next(iterator)
    assert feature_processor_segment_names == ["jan2", "feb2"]
    assert segment_name == "feb"
    assert list(data.columns) == ["c", "d", "weight"]
    assert data.shape == (256, 3)
    assert data.sum().sum() == 1024.0
