import pytest
from eemeter.ee.derivatives import DerivativePair, Derivative
from eemeter.ee.aggregate import Aggregator


@pytest.fixture
def derivative_pairs_empty():
    return []


@pytest.fixture
def derivative_pairs_single():
    return [
        DerivativePair(
            "interpretation",
            "kWh",
            Derivative("1", 10, 3, 3, 5, None),
            Derivative("2", 10, 6, 6, 5, None),
        ),
    ]


@pytest.fixture
def derivative_pairs():
    return [
        DerivativePair(
            "interpretation",
            "kWh",
            Derivative("1", 10, 3, 3, 5, None),
            Derivative("2", 10, 6, 6, 5, None),
        ),
        DerivativePair(
            "interpretation",
            "kWh",
            Derivative("1", 10, 4, 4, 5, None),
            Derivative("2", 10, 8, 8, 5, None),
        ),
    ]


@pytest.fixture
def derivative_pairs_mixed_interpretation():
    return [
        DerivativePair(
            "interpretation1",
            "kWh",
            Derivative("1", 10, 3, 3, 5, None),
            Derivative("2", 10, 6, 6, 5, None),
        ),
        DerivativePair(
            "interpretation2",
            "kWh",
            Derivative("1", 10, 4, 4, 5, None),
            Derivative("2", 10, 8, 8, 5, None),
        ),
    ]

@pytest.fixture
def derivative_pairs_mixed_unit():
    return [
        DerivativePair(
            "interpretation1",
            "kWh",
            Derivative("1", 10, 3, 3, 5, None),
            Derivative("2", 10, 6, 6, 5, None),
        ),
        DerivativePair(
            "interpretation2",
            "therm",
            Derivative("1", 10, 4, 4, 5, None),
            Derivative("2", 10, 8, 8, 5, None),
        ),
    ]


@pytest.fixture
def derivative_pairs_one_invalid():
    return [
        DerivativePair(
            "interpretation",
            "kWh",
            Derivative("1", None, None, None, None, None),
            Derivative("2", 10, 3, 3, 5, None),
        ),
        DerivativePair(
            "interpretation",
            "kWh",
            Derivative("1", 10, 4, 4, 5, None),
            Derivative("2", 10, 8, 8, 5, None),
        ),
    ]


def test_basic_usage(derivative_pairs):

    aggregator = Aggregator("SUM")
    derivative_pair, n_valid, n_invalid = \
        aggregator.aggregate(derivative_pairs, "interpretation")

    assert n_valid == 2
    assert n_invalid == 0

    assert derivative_pair.interpretation == "interpretation"

    baseline = derivative_pair.baseline
    assert baseline.label is None
    assert baseline.value == 20
    assert baseline.lower == 5
    assert baseline.upper == 5
    assert baseline.n == 10

    reporting = derivative_pair.reporting
    assert reporting.label is None
    assert reporting.value == 20
    assert reporting.lower == 10
    assert reporting.upper == 10
    assert reporting.n == 10

def test_empty(derivative_pairs_empty):
    aggregator = Aggregator()
    with pytest.raises(ValueError):
        aggregator.aggregate(derivative_pairs_empty)


def test_single(derivative_pairs_single):
    aggregator = Aggregator()

    derivative_pairs, n_valid, n_invalid = \
        aggregator.aggregate(derivative_pairs_single)

    assert n_valid == 1
    assert n_invalid == 0


def test_mixed_interpretaiton_fails(derivative_pairs_mixed_interpretation):

    aggregator = Aggregator()

    with pytest.raises(ValueError):
        derivative_pair, n_valid, n_invalid = \
            aggregator.aggregate(derivative_pairs_mixed_interpretation,
                                 "interpretation1")


def test_mixed_unit_fails(derivative_pairs_mixed_unit):

    aggregator = Aggregator()

    with pytest.raises(ValueError):
        derivative_pair, n_valid, n_invalid = \
            aggregator.aggregate(derivative_pairs_mixed_unit,
                                 "kWh")


def test_missing(derivative_pairs_one_invalid):

    aggregator = Aggregator()

    derivative_pair, n_valid, n_invalid = \
        aggregator.aggregate(derivative_pairs_one_invalid,
                             "interpretation")

    assert n_valid == 1
    assert n_invalid == 1


def test_missing_with_default(derivative_pairs_one_invalid):

    aggregator = Aggregator(baseline_default_value=Derivative(
        None, 0, 0, 0, 0, None
    ))

    derivative_pair, n_valid, n_invalid = \
        aggregator.aggregate(derivative_pairs_one_invalid,
                             "interpretation")

    assert n_valid == 2
    assert n_invalid == 0
