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
import numpy as np

from sqlalchemy import create_engine
from sqlalchemy import Table, Column, Integer, Float, Numeric, String, MetaData, ForeignKey, TIMESTAMP

RTOL = 1e-2
ATOL = 1e-2

def test_import_hpxml(consumption_hpxml_filename):
    consumption = import_hpxml(consumption_hpxml_filename)

    elec_data = consumption[0].data
    gas_data = consumption[1].data
    elec_estimated = consumption[0].estimated
    gas_estimated = consumption[1].estimated
    assert_allclose(elec_data.values, [10.,11.,12., np.nan],
            rtol=RTOL, atol=ATOL)
    assert_allclose(gas_data.values, [10.,11.,12., np.nan],
            rtol=RTOL, atol=ATOL)
    assert_allclose(elec_estimated.values, [False, False, True, False],
            rtol=RTOL, atol=ATOL)
    assert_allclose(gas_estimated.values, [False, False, True, False],
            rtol=RTOL, atol=ATOL)

def test_import_green_button_xml(consumption_gbxml_filename):
    cd = import_green_button_xml(consumption_gbxml_filename)

    assert_allclose(cd.data.values,
            [25.662, 21.021, 21.021, 21.021, 21.021,
             21.021, 25.662, 25.662, 21.021, 21.021,
             21.021, 21.021, 21.021, 25.662, 25.662,
             21.021, 21.021, 21.021, 21.021, 21.021,
             25.662, 25.662, 21.021, 21.021, 21.021,
             21.021, 21.021, 25.662, 25.662, 21.021,
             21.021, np.nan, 25.662, 25.662, 21.021,
             21.021, 21.021, 21.021, 21.021, 25.662,
             25.389, 21.021, 21.021, 21.021, 21.021,
             21.021, 25.662, 25.662, 21.021, 21.021,
             21.021, 21.021, np.nan], rtol=RTOL, atol=ATOL)

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

    building_1_consumption = buildings[1]
    building_2_consumption = buildings[2]

    c_1_e = building_1_consumption[0]
    c_1_g = building_1_consumption[1]
    c_2_e = building_2_consumption[0]
    c_2_g = building_2_consumption[1]


    assert_allclose(c_1_e.data.values,[1,1,1,1,np.nan],rtol=RTOL,atol=ATOL)
    assert_allclose(c_1_g.data.values,[1,1,1,1,np.nan],rtol=RTOL,atol=ATOL)
    assert_allclose(c_2_e.data.values,[1,1,1,1,np.nan],rtol=RTOL,atol=ATOL)
    assert_allclose(c_2_g.data.values,[1,1,1,1,np.nan],rtol=RTOL,atol=ATOL)

    assert c_1_e.data.index[0] == datetime(2011,1,1)
    assert c_1_g.data.index[4] == datetime(2011,5,1)

def test_import_csv(consumption_csv_filename):

    cd = import_csv(consumption_csv_filename)

    assert_allclose(cd.data.values,[25,np.nan],rtol=RTOL,atol=ATOL)
    assert_allclose(cd.estimated.values,[True,False],rtol=RTOL,atol=ATOL)

def test_import_pandas():
    df = pd.DataFrame({"Consumption": [25,1000],
                       "UnitofMeasure": ["therms","kWh"],
                       "FuelType":["natural gas","electricity"],
                       "StartDateTime":[datetime(2013,12,15),datetime(2013,11,10)],
                       "EndDateTime":[datetime(2014,1,14),datetime(2013,12,15)],
                       "ReadingType":["estimated","actual"]})

    cd = import_pandas(df)

    assert_allclose(cd.data.values,[25,np.nan],rtol=RTOL,atol=ATOL)
    assert_allclose(cd.estimated.values,[True,False],rtol=RTOL,atol=ATOL)

def test_import_excel(consumption_xlsx_filename):

    cd = import_excel(consumption_xlsx_filename)

    assert_allclose(cd.data.values,[25,np.nan],rtol=RTOL,atol=ATOL)
    assert_allclose(cd.estimated.values,[True,False],rtol=RTOL,atol=ATOL)
