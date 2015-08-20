from eemeter.models.parameters import ParameterType

import numpy as np

import pytest

def test_parameter_type():
    class TestParameters(ParameterType):
        parameters = ["param1", "param2"]

    def check_vals(tp):
        assert tp.to_list() == [0,1]
        assert all(tp.to_array() == np.array([0,1]))
        assert tp.to_dict()["param1"] == 0
        assert tp.to_dict()["param2"] == 1
        assert len(tp.to_dict()) == 2

    check_vals(TestParameters([0,1]))

    check_vals(TestParameters(np.array([0,1])))

    check_vals(TestParameters({"param1": 0, "param2": 1}))

    # too few params
    with pytest.raises(TypeError) as e:
        TestParameters([])

    # too many params
    with pytest.raises(TypeError) as e:
        TestParameters([0,0,0])

    # too few params
    with pytest.raises(TypeError) as e:
        TestParameters({})

    # wrong params
    with pytest.raises(KeyError) as e:
        TestParameters({"wrong1": 0,"wrong2": 0})

    # wrong value type
    with pytest.raises(TypeError) as e:
        TestParameters(0)
