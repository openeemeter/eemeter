from eemeter.structures import EnergyTrace
from eemeter.io.serializers import ArbitrarySerializer
import pandas as pd
import numpy as np
from datetime import datetime
import pytz

import pytest


@pytest.fixture
def interpretation():
    return 'ELECTRICITY_CONSUMPTION_SUPPLIED'


def test_no_data_no_placeholder(interpretation):
    with pytest.raises(ValueError):
        EnergyTrace(interpretation=interpretation)


def test_data_and_placeholder(interpretation):
    with pytest.raises(ValueError):
        EnergyTrace(interpretation=interpretation, data=pd.DataFrame(),
                    placeholder=True)


def test_placeholder_valid(interpretation):
    et = EnergyTrace(interpretation=interpretation, placeholder=True)

    assert et.interpretation == interpretation
    assert et.data is None
    assert et.unit is None
    assert et.placeholder


def test_invalid_interpretation():
    with pytest.raises(ValueError):
        EnergyTrace(interpretation="INVALID", placeholder=True)


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
        EnergyTrace(interpretation=interpretation, data=pd.DataFrame())


def test_data_but_invalid_unit(interpretation):
    with pytest.raises(ValueError):
        EnergyTrace(interpretation=interpretation, data=pd.DataFrame(),
                    unit="INVALID")


def test_has_trace_id_and_interval():
    et = EnergyTrace(
        interpretation='ELECTRICITY_CONSUMPTION_SUPPLIED',
        placeholder=True
    )
    assert et.trace_id is None
    assert et.interval is None

    et = EnergyTrace(
        interpretation='ELECTRICITY_CONSUMPTION_SUPPLIED',
        placeholder=True,
        trace_id='ABC',
        interval='daily'
    )
    assert et.trace_id == 'ABC'
    assert et.interval == 'daily'

    assert 'ABC' in str(et)


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
    data = {"value": [1, np.nan], "estimated": [False, False]}
    columns = ["value", "estimated"]
    index = pd.date_range('2000-01-01', periods=2, freq='D')
    return pd.DataFrame(data, index=index, columns=columns)


def test_data_and_valid_unit(
        interpretation, unnormalized_unit_with_target_unit, unit_timeseries):

    unnormalized_unit, normalized_unit, mult = \
        unnormalized_unit_with_target_unit

    et = EnergyTrace(interpretation=interpretation, data=unit_timeseries,
                     unit=unnormalized_unit)
    assert et.interpretation == interpretation
    assert et.unit == normalized_unit
    np.testing.assert_allclose(
        et.data.value.iloc[0], (unit_timeseries.value * mult).iloc[0],
        rtol=1e-3, atol=1e-3)
    assert not et.data.estimated.iloc[0]
    assert not et.placeholder


@pytest.fixture
def serializer():
    return ArbitrarySerializer()


@pytest.fixture
def records():
    return [{
        'start': datetime(2000, 1, 1, tzinfo=pytz.UTC),
        'end': datetime(2000, 1, 2, tzinfo=pytz.UTC),
        'value': 1,
    }]


def test_serializer(interpretation, records, unit, serializer):

    et = EnergyTrace(interpretation=interpretation, records=records, unit=unit,
                     serializer=serializer)

    assert et.data.value.iloc[0] == records[0]['value']
    assert not et.data.estimated.iloc[0]


def test_non_timeseries_data(interpretation, unit):

    data = {"value": [1, np.nan], "estimated": [False, False]}
    columns = ["value", "estimated"]

    df = pd.DataFrame(data, columns=columns)

    with pytest.raises(ValueError):
        EnergyTrace(interpretation=interpretation, data=df, unit=unit)


def test_bad_column_name_data(interpretation, unit):

    data = {"energy": [1, np.nan], "estimated": [False, False]}
    columns = ["energy", "estimated"]
    index = pd.date_range('2000-01-01', periods=2, freq='D')

    df = pd.DataFrame(data, index=index, columns=columns)

    with pytest.raises(ValueError):
        EnergyTrace(interpretation=interpretation, data=df, unit=unit)


def test_repr(interpretation):
    et = EnergyTrace(interpretation=interpretation, placeholder=True)
    assert 'EnergyTrace' in str(et)
