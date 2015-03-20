from eemeter.config.yaml_parser import load

from eemeter.meter import Sequence
from eemeter.meter import Condition
from eemeter.meter import And
from eemeter.meter import Or

import pytest

def test_sequential_meter():
    meter_yaml = """
        !obj:eemeter.meter.Sequence {
            sequence: [
                !obj:eemeter.meter.DummyMeter {},
            ]
        }"""

    meter = load(meter_yaml)

    result = meter.evaluate(value=10)

    assert result["result"] == 10

def test_conditional_meter():
    meter_yaml="""
        !obj:eemeter.meter.Condition {
            condition_parameter: "electricity_present",
            success: !obj:eemeter.meter.DummyMeter {
                input_mapping: {"success":"value"},
            },
            failure: !obj:eemeter.meter.DummyMeter {
                input_mapping: {"failure":"value"},
            },
        }
        """
    meter = load(meter_yaml)
    assert meter.evaluate(electricity_present=True,success="success",failure="failure")["result"] == "success"
    assert meter.evaluate(electricity_present=False,success="success",failure="failure")["result"] == "failure"

def test_conditional_meter_without_params():
    meter_yaml="""
        !obj:eemeter.meter.Condition {
            condition_parameter: "electricity_present",
        }
        """
    meter = load(meter_yaml)
    assert isinstance(meter.evaluate(electricity_present=True),dict)
    assert isinstance(meter.evaluate(electricity_present=False),dict)

def test_and_meter():
    with pytest.raises(ValueError):
        meter0 = And(inputs=[])

    meter1 = And(inputs=["result_one"])
    assert meter1.evaluate(result_one=True,result_two=True)["output"]

    meter2 = And(inputs=["result_one","result_two"])
    assert meter2.evaluate(result_one=True,result_two=True)["output"]
    assert not meter2.evaluate(result_one=False,result_two=True)["output"]
    assert not meter2.evaluate(result_one=True,result_two=False)["output"]
    assert not meter2.evaluate(result_one=False,result_two=False)["output"]

    meter3 = And(inputs=["result_one","result_two","result_three"])
    with pytest.raises(ValueError):
        assert meter3.evaluate(result_one=True,result_two=True)
    assert meter3.evaluate(result_one=True,result_two=True,result_three=True)["output"]
    assert not meter3.evaluate(result_one=True,result_two=True,result_three=False)["output"]
    assert not meter3.evaluate(result_one=False,result_two=True,result_three=True)["output"]


def test_or_meter():
    with pytest.raises(ValueError):
        meter0 = Or(inputs=[])

    meter1 = Or(inputs=["result_one"])
    assert meter1.evaluate(result_one=True,result_two=True)["output"]

    meter2 = Or(inputs=["result_one","result_two"])
    assert meter2.evaluate(result_one=True,result_two=True)["output"]
    assert meter2.evaluate(result_one=False,result_two=True)["output"]
    assert meter2.evaluate(result_one=True,result_two=False)["output"]
    assert not meter2.evaluate(result_one=False,result_two=False)["output"]

    meter3 = Or(inputs=["result_one","result_two","result_three"])
    with pytest.raises(ValueError):
        assert meter3.evaluate(result_one=True,result_two=True)
    assert meter3.evaluate(result_one=True,result_two=True,result_three=True)["output"]
    assert meter3.evaluate(result_one=True,result_two=True,result_three=False)["output"]
    assert meter3.evaluate(result_one=False,result_two=True,result_three=True)["output"]
    assert not meter3.evaluate(result_one=False,result_two=False,result_three=False)["output"]

def test_switch():
    meter_yaml = """
        !obj:eemeter.meter.Switch {
            target: target,
            cases: {
                1: !obj:eemeter.meter.DummyMeter {
                    input_mapping: {
                        value_one: value,
                    },
                },
                2: !obj:eemeter.meter.DummyMeter {
                    input_mapping: {
                        value_two: value,
                    },
                },
                3: !obj:eemeter.meter.DummyMeter {
                    input_mapping: {
                        value_three: value,
                    },
                },
            },
            default: !obj:eemeter.meter.DummyMeter {
                input_mapping: {
                    value_default: value,
                },
            }
        }
    """

    meter = load(meter_yaml)

    result1 = meter.evaluate(target=1,value_one=1,value_two=2,value_three=3,value_default=4)
    result2 = meter.evaluate(target=2,value_one=1,value_two=2,value_three=3,value_default=4)
    result3 = meter.evaluate(target=3,value_one=1,value_two=2,value_three=3,value_default=4)
    result4 = meter.evaluate(target=4,value_one=1,value_two=2,value_three=3,value_default=4)
    result5 = meter.evaluate(value_one=1,value_two=2,value_three=3,value_default=4)

    assert 1 == result1.get("result")
    assert 2 == result2.get("result")
    assert 3 == result3.get("result")
    assert 4 == result4.get("result")
    assert None == result5.get("result")
