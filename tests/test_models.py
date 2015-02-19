from eemeter.models import HDDCDDBalancePointModel
from eemeter.models import HDDBalancePointModel
from eemeter.models import CDDBalancePointModel

from numpy.testing import assert_almost_equal

import pytest

def test_HDDCDDBalancePointModel():
    x0 = [0,0,0,55,2]
    bounds = [[0,100],[0,100],[0,100],[50,60],[2,12]]
    model = HDDCDDBalancePointModel()
    model.x0 = None
    model.bounds = None
    model = HDDCDDBalancePointModel(x0,bounds)
    model = HDDCDDBalancePointModel(x0=x0,bounds=bounds)

    params = [1,1,1,60,5]
    observed_temps = [[i] for i in range(50,70)]
    usages = model.compute_usage_estimates(params,observed_temps)
    assert_almost_equal(usages[8:18],[3,2,1,1,1,1,1,1,2,3])
    opt_params = model.parameter_optimization(usages, observed_temps)
    assert_almost_equal(params,opt_params,decimal=3)


def test_HDDBalancePointModel():
    x0 = [55,0,0]
    bounds = [[55,65],[0,100],[0,100]]
    model = HDDBalancePointModel()
    model.x0 = None
    model.bounds = None
    model = HDDBalancePointModel(x0,bounds)
    model = HDDBalancePointModel(x0=x0,bounds=bounds)

    params = [60,1,1]
    observed_temps = [[i] for i in range(50,70)]
    usages = model.compute_usage_estimates(params,observed_temps)
    assert_almost_equal(usages[8:13],[3,2,1,1,1])
    opt_params = model.parameter_optimization(usages, observed_temps)
    assert_almost_equal(params,opt_params,decimal=3)

def test_CDDBalancePointModel():
    x0 = [55,0,0]
    bounds = [[55,65],[0,100],[0,100]]
    model = CDDBalancePointModel()
    model.x0 = None
    model.bounds = None
    model = CDDBalancePointModel(x0,bounds)
    model = CDDBalancePointModel(x0=x0,bounds=bounds)

    params = [60,1,1]
    observed_temps = [[i] for i in range(50,70)]
    usages = model.compute_usage_estimates(params,observed_temps)
    assert_almost_equal(usages[8:13],[1,1,1,2,3])
    opt_params = model.parameter_optimization(usages, observed_temps)
    assert_almost_equal(params,opt_params,decimal=3)
