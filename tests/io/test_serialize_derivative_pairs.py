import json

import pytest

from eemeter.io.serializers.meter_output import serialize_derivative_pairs
from eemeter.ee.derivatives import DerivativePair, Derivative


@pytest.fixture
def derivative_pairs():
    return [
        DerivativePair(
            "interpretation1",
            Derivative("1", 10, 3, 3, 5, None),
            Derivative("2", 10, 6, 6, 5, None),
        ),
        DerivativePair(
            "interpretation2",
            None,
            Derivative("2", 10, 8, 8, 5, None),
        ),
    ]


def test_basic_usage(derivative_pairs):
    serialized = serialize_derivative_pairs(derivative_pairs)
    assert len(serialized) == 2
    assert len(json.dumps(serialized)) == 317
    assert serialized[0]["interpretation"] == "interpretation1"
    assert serialized[1]["interpretation"] == "interpretation2"

    assert serialized[0]["baseline"]["value"] == 10
    assert serialized[0]["reporting"]["value"] == 10
    assert serialized[1]["baseline"] is None
    assert serialized[1]["reporting"]["value"] == 10
