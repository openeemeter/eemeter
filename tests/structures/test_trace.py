from eemeter.structures import EnergyTrace
import pandas as pd
import numpy as np
import pytest


@pytest.fixture
def interpretation():
    return 'ELECTRICITY_CONSUMPTION_SUPPLIED'

def test_no_data_no_placeholder(interpretation):
    with pytest.raises(ValueError):
        et = EnergyTrace(interpretation=interpretation)

def test_data_and_placeholder(interpretation):
    with pytest.raises(ValueError):
        et = EnergyTrace(interpretation=interpretation, data=pd.DataFrame(),
                         placeholder=True)

def test_placeholder_valid(interpretation):
    et = EnergyTrace(interpretation=interpretation, placeholder=True)

    assert et.interpretation == interpretation
    assert et.data is None
    assert et.unit is None
    assert et.placeholder == True

def test_invalid_interpretation():
    with pytest.raises(ValueError):
        et = EnergyTrace(interpretation="INVALID", placeholder=True)

@pytest.fixture(params=[
    'ELECTRICITY_CONSUMPTION_SUPPLIED',
    'ELECTRICITY_CONSUMPTION_TOTAL',
    'ELECTRICITY_CONSUMPTION_NET',
    'ELECTRICITY_ON_SITE_GENERATION_TOTAL',
    'ELECTRICITY_ON_SITE_GENERATION_CONSUMED',
    'ELECTRICITY_ON_SITE_GENERATION_UNCONSUMED',
    'NATURAL_GAS_CONSUMPTION_SUPPLIED',
])
def valid_interpretation(request):
    return request.param

def test_valid_interpretation(valid_interpretation):
    et = EnergyTrace(interpretation=valid_interpretation, placeholder=True)

    assert et.interpretation == valid_interpretation

def test_data_but_no_unit(interpretation):
    with pytest.raises(ValueError):
        et = EnergyTrace(interpretation=interpretation, data=pd.DataFrame())

def test_data_but_invalid_unit(interpretation):
    with pytest.raises(ValueError):
        et = EnergyTrace(interpretation=interpretation, data=pd.DataFrame(),
                         unit="INVALID")

@pytest.fixture
def unit():
    return "KWH"

@pytest.fixture(params=[
    ('wh', 'KWH', 0.001),
    ('Wh', 'KWH', 0.001),
    ('WH', 'KWH', 0.001),
    ('kwh', 'KWH', 1),
    ('kWh', 'KWH', 1),
    ('KWH', 'KWH', 1),
    ('thm', 'THERM', 1),
    ('THM', 'THERM', 1),
    ('therm', 'THERM', 1),
    ('THERM', 'THERM', 1),
    ('therms', 'THERM', 1),
    ('THERMS', 'THERM', 1),
])
def unnormalized_unit_with_target_unit(request):
    return request.param

@pytest.fixture
def unit_timeseries():
    data = {"energy": [1, np.nan], "estimated": [False, False]}
    columns = ["energy", "estimated"]
    index = pd.date_range('2000-01-01', periods=2, freq='D')
    return pd.DataFrame(data, index=index, columns=columns)

def test_data_and_valid_unit(
        interpretation, unnormalized_unit_with_target_unit, unit_timeseries):

    unnormalized_unit, normalized_unit, mult = \
        unnormalized_unit_with_target_unit

    et = EnergyTrace(interpretation=interpretation, data=unit_timeseries,
                     unit=unnormalized_unit)
    assert et.unit == normalized_unit
    assert et.data.energy.iloc[0] == (unit_timeseries.energy * mult).iloc[0]
