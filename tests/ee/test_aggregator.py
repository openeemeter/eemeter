import pytest
from eemeter.ee.meter import DerivativePair, Derivative, Aggregator


@pytest.fixture
def derivative_pairs():
    return [
        DerivativePair(
            "interpretation",
            Derivative("1", 10, 3, 3, 5),
            Derivative("2", 10, 6, 6, 5),
        ),
        DerivativePair(
            "interpretation",
            Derivative("1", 10, 4, 4, 5),
            Derivative("2", 10, 8, 8, 5),
        ),
    ]


def test_basic_usage(derivative_pairs):

    def sum_func(d1, d2):
        return Derivative(
            None,
            d1.value + d2.value,
            (d1.lower**2 + d2.lower**2)**0.5,
            (d1.upper**2 + d2.upper**2)**0.5,
            d1.n + d2.n,
        )

    aggregator = Aggregator(sum_func)
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
