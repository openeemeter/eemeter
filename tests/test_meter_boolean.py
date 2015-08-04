from eemeter.meter import And
from eemeter.meter import Or

import pytest

def test_and_meter():
    with pytest.raises(ValueError):
        meter0 = And(inputs=[])

    meter1 = And(inputs=["result_one"])
    assert meter1.evaluate_raw(result_one=True, result_two=True)["output"]

    meter2 = And(inputs=["result_one", "result_two"])
    assert meter2.evaluate_raw(result_one=True, result_two=True)["output"]
    assert not meter2.evaluate_raw(result_one=False, result_two=True)["output"]
    assert not meter2.evaluate_raw(result_one=True, result_two=False)["output"]
    assert not meter2.evaluate_raw(result_one=False,
            result_two=False)["output"]

    meter3 = And(inputs=["result_one", "result_two", "result_three"])
    with pytest.raises(ValueError):
        assert meter3.evaluate_raw(result_one=True, result_two=True)
    assert meter3.evaluate_raw(result_one=True, result_two=True,
            result_three=True)["output"]
    assert not meter3.evaluate_raw(result_one=True, result_two=True,
            result_three=False)["output"]
    assert not meter3.evaluate_raw(result_one=False, result_two=True,
            result_three=True)["output"]


def test_or_meter():
    with pytest.raises(ValueError):
        meter0 = Or(inputs=[])

    meter1 = Or(inputs=["result_one"])
    assert meter1.evaluate_raw(result_one=True, result_two=True)["output"]

    meter2 = Or(inputs=["result_one", "result_two"])
    assert meter2.evaluate_raw(result_one=True, result_two=True)["output"]
    assert meter2.evaluate_raw(result_one=False, result_two=True)["output"]
    assert meter2.evaluate_raw(result_one=True, result_two=False)["output"]
    assert not meter2.evaluate_raw(result_one=False,
            result_two=False)["output"]

    meter3 = Or(inputs=["result_one", "result_two", "result_three"])
    with pytest.raises(ValueError):
        assert meter3.evaluate_raw(result_one=True, result_two=True)
    assert meter3.evaluate_raw(result_one=True, result_two=True,
            result_three=True)["output"]
    assert meter3.evaluate_raw(result_one=True, result_two=True,
            result_three=False)["output"]
    assert meter3.evaluate_raw(result_one=False, result_two=True,
            result_three=True)["output"]
    assert not meter3.evaluate_raw(result_one=False, result_two=False,
            result_three=False)["output"]
