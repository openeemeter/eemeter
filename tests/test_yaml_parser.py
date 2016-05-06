from eemeter.config.yaml_parser import load_path,load,dump

import yaml
import tempfile
import os
from decimal import Decimal
from eemeter.config.yaml_parser import Setting

from eemeter.meter import BPI_2400_S_2012_ModelCalibrationUtilityBillCriteria
from eemeter.meter import DefaultResidentialMeter

import pytest

@pytest.fixture(params=["!obj:decimal.Decimal { value : '1.23' }",
                        "!obj:decimal.Decimal {'value' : '1.23'}",
                        """!obj:decimal.Decimal
                              value: '1.23'"""])
def simple_yaml(request):
    return request.param

def test_load_path():
    fd, fname = tempfile.mkstemp()
    with os.fdopen(fd, 'wb') as f:
        f.write('a: 23'.encode('utf-8'))
    loaded = load_path(fname)
    assert loaded['a'] == 23
    os.remove(fname)

def test_obj():
    loaded1 = load("a: !obj:decimal.Decimal { value : '1.23' }")
    assert isinstance(loaded1['a'], Decimal)

def test_obj_formats(simple_yaml):
    loaded = load(simple_yaml)
    assert isinstance(loaded, Decimal)

def test_setting():
    settings = {"heating_config": 10}

    loaded = load("a: !setting heating_config", settings=settings)
    assert loaded['a'] == 10

    # no settings provided
    with pytest.raises(KeyError):
        loaded = load("a: !setting heating_config")

def test_dump_meter():

    # just make sure nothing has changed unexpectedly
    meter = BPI_2400_S_2012_ModelCalibrationUtilityBillCriteria("degF")
    assert len(dump(meter.meter)) == 11913

    meter = DefaultResidentialMeter("degF")
    assert len(dump(meter.meter)) == 21655

