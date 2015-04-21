import tempfile
import os
from eemeter.importers import import_hpxml
from eemeter.importers import import_green_button_xml
from eemeter.importers import import_seed_timeseries
from eemeter.importers import import_csv
from eemeter.importers import import_pandas
from eemeter.importers import import_excel

from fixtures.importers import consumption_csv_filename
from fixtures.importers import consumption_xlsx_filename
from fixtures.importers import consumption_hpxml_filename
from fixtures.importers import consumption_gbxml_filename

from numpy.testing import assert_allclose
from datetime import datetime
from datetime import timedelta

import pandas as pd

from sqlalchemy import create_engine
from sqlalchemy import Table, Column, Integer, Float, Numeric, String, MetaData, ForeignKey, TIMESTAMP

RTOL = 1e-2
ATOL = 1e-2

def test_import_hpxml(consumption_hpxml_filename):
    ch = import_hpxml(consumption_hpxml_filename)

    consumptions_e = ch.electricity
    consumptions_g = ch.natural_gas
    assert_allclose([ c.kWh for c in consumptions_e],[10.,11.,12.],rtol=RTOL,atol=ATOL)
    assert_allclose([ c.therms for c in consumptions_g],[10.,11.,12.],rtol=RTOL,atol=ATOL)
    assert len(consumptions_e) == 3
    assert len(consumptions_g) == 3
    assert not consumptions_e[0].estimated
    assert not consumptions_e[1].estimated
    assert not consumptions_g[0].estimated
    assert not consumptions_g[1].estimated
    assert consumptions_e[2].estimated
    assert consumptions_g[2].estimated

def test_import_green_button_xml(consumption_gbxml_filename):
    ch = import_green_button_xml(consumption_gbxml_filename)

    consumptions_e = ch.electricity

    c_kWh = [c.kWh for c in consumptions_e]

    assert_allclose(c_kWh, [25.662, 21.021, 21.021, 21.021, 21.021,
                            21.021, 25.662, 25.662, 21.021, 21.021,
                            21.021, 21.021, 21.021, 25.662, 25.662,
                            21.021, 21.021, 21.021, 21.021, 21.021,
                            25.662, 25.662, 21.021, 21.021, 21.021,
                            21.021, 21.021, 25.662, 25.662, 21.021,
                            21.021, 25.662, 25.662, 21.021, 21.021,
                            21.021, 21.021, 21.021, 25.662, 25.389,
                            21.021, 21.021, 21.021, 21.021, 21.021,
                            25.662, 25.662, 21.021, 21.021, 21.021,
                            21.021], rtol=RTOL, atol=ATOL)

def test_import_seed_timeseries():
    fd, fname = tempfile.mkstemp()
    db_url = 'sqlite:///{}'.format(fname)
    engine = create_engine(db_url, echo=True)
    metadata = MetaData()
    seed_meter = Table('seed_meter', metadata,
        Column('id', Integer, primary_key=True),
        Column('name', String),
        Column('energy_type', Integer),
        Column('energy_units', Integer),
    )
    seed_timeseries = Table('seed_timeseries', metadata,
        Column('id', Integer, primary_key=True),
        Column('meter_id', None, ForeignKey('seed_meter.id')),
        Column('reading', Float),
        Column('cost', Numeric),
        Column('begin_time', TIMESTAMP),
        Column('end_time', TIMESTAMP),
    )
    seed_meter_building_snapshot = Table('seed_meter_building_snapshot', metadata,
        Column('id', Integer, primary_key=True),
        Column('meter_id', None, ForeignKey('seed_meter.id')),
        Column('buildingsnapshot_id', Integer),
    )
    metadata.create_all(engine)

    conn = engine.connect()

    conn.execute(seed_meter.insert(), [
        {"name": "test1", "energy_type": 2, "energy_units": 1},
        {"name": "test2", "energy_type": 1, "energy_units": 2},
        {"name": "test3", "energy_type": 2, "energy_units": 1},
        {"name": "test4", "energy_type": 1, "energy_units": 2},
    ])

    dates = [ datetime(2011,1,1) + timedelta(days=i) for i in range(0,120,30)]

    timestamps = [(d, d + timedelta(days=30)) for d in dates]

    conn.execute(seed_timeseries.insert(), [
        {"meter_id": meter, "reading": 1.0, "cost": 0, "begin_time": ts[0],"end_time": ts[1]}
        for ts in timestamps for meter in [1,2,3,4]]
    )

    conn.execute(seed_meter_building_snapshot.insert(), [
        {"meter_id": 1, "buildingsnapshot_id": 1},
        {"meter_id": 2, "buildingsnapshot_id": 1},
        {"meter_id": 3, "buildingsnapshot_id": 2},
        {"meter_id": 4, "buildingsnapshot_id": 2},
    ])

    buildings = import_seed_timeseries(db_url)

    building_1_ch = buildings[1]
    building_2_ch = buildings[2]

    c_1_e = building_1_ch.electricity
    c_1_g = building_1_ch.natural_gas
    c_2_e = building_2_ch.electricity
    c_2_g = building_2_ch.natural_gas


    assert_allclose([c.kWh for c in c_1_e],[1,1,1,1],rtol=RTOL,atol=ATOL)
    assert_allclose([c.therms for c in c_1_g],[1,1,1,1],rtol=RTOL,atol=ATOL)
    assert_allclose([c.kWh for c in c_2_e],[1,1,1,1],rtol=RTOL,atol=ATOL)
    assert_allclose([c.therms for c in c_2_g],[1,1,1,1],rtol=RTOL,atol=ATOL)

    assert c_1_e[0].start == datetime(2011,1,1)
    assert c_1_g[3].end == datetime(2011,5,1)

def test_import_csv(consumption_csv_filename):

    ch = import_csv(consumption_csv_filename)

    assert len(ch.natural_gas) == 1
    assert len(ch.electricity) == 1
    assert_allclose(ch.natural_gas[0].therms,25,rtol=RTOL,atol=ATOL)
    assert_allclose(ch.electricity[0].kWh,1000,rtol=RTOL,atol=ATOL)
    assert ch.natural_gas[0].estimated
    assert not ch.electricity[0].estimated
    assert ch.natural_gas[0].timedelta.days == 30
    assert ch.electricity[0].timedelta.days == 35

def test_import_pandas():
    df = pd.DataFrame({"Consumption": [25,1000],
                       "UnitofMeasure": ["therms","kWh"],
                       "FuelType":["natural gas","electricity"],
                       "StartDateTime":[datetime(2013,12,15),datetime(2013,11,10)],
                       "EndDateTime":[datetime(2014,1,14),datetime(2013,12,15)],
                       "ReadingType":["estimated","actual"]})

    ch = import_pandas(df)

    assert len(ch.natural_gas) == 1
    assert len(ch.electricity) == 1
    assert_allclose(ch.natural_gas[0].therms,25,rtol=RTOL,atol=ATOL)
    assert_allclose(ch.electricity[0].kWh,1000,rtol=RTOL,atol=ATOL)
    assert ch.natural_gas[0].estimated
    assert not ch.electricity[0].estimated
    assert ch.natural_gas[0].timedelta.days == 30
    assert ch.electricity[0].timedelta.days == 35

def test_import_excel(consumption_xlsx_filename):

    ch = import_excel(consumption_xlsx_filename)

    assert len(ch.natural_gas) == 1
    assert len(ch.electricity) == 1
    assert_allclose(ch.natural_gas[0].therms,25,rtol=RTOL,atol=ATOL)
    assert_allclose(ch.electricity[0].kWh,1000,rtol=RTOL,atol=ATOL)
    assert ch.natural_gas[0].estimated
    assert not ch.electricity[0].estimated
    assert ch.natural_gas[0].timedelta.days == 30
    assert ch.electricity[0].timedelta.days == 35
