from lxml import etree
from dateutil.parser import parse
from eemeter.consumption import ConsumptionData
from datetime import datetime
from csv import DictReader
import dateutil.parser
from warnings import warn

import pandas as pd

from sqlalchemy import create_engine
from sqlalchemy import Table, Column, Integer, Float, Numeric, String, MetaData, ForeignKey, TIMESTAMP
from sqlalchemy.sql import select


### TODO: DEPRECATE THESE --->> replacing in eemeter.parsers

def import_hpxml(filename):
    """Import from HPXML 2.0.

    Parameters
    ----------
    filename : str
        Full path to HPXML file

    Returns
    -------
    out : list of eemeter.consumption.ConsumptionData
        Consumption data available for this project
    """

    hpxml_fuel_type_mapping = {
        "electricity": "electricity",
        "natural gas": "natural_gas",
    }

    hpxml_unit_of_measure_mapping = {
        "kWh": "kWh",
        "therms": "therm",
    }

    with(open(filename,'r')) as f:
        tree = etree.parse(f)

    root = tree.getroot()
    ns = root.nsmap

    consumption_info_xpath = "ns2:Consumption/ns2:ConsumptionDetails/ns2:ConsumptionInfo"
    consumption_infos = root.xpath(consumption_info_xpath,namespaces=ns)

    consumption = []
    for info in consumption_infos:
        fuel_type = info.xpath("ns2:ConsumptionType/ns2:Energy/ns2:FuelType",namespaces=ns)
        unit_of_measure = info.xpath("ns2:ConsumptionType/ns2:Energy/ns2:UnitofMeasure",namespaces=ns)
        if fuel_type == [] or unit_of_measure == []:
            continue
        fuel_type = hpxml_fuel_type_mapping[fuel_type[0].text]
        unit_str = hpxml_unit_of_measure_mapping[unit_of_measure[0].text]
        consumption_details = info.xpath("ns2:ConsumptionDetail",namespaces=ns)
        records = []
        for details in consumption_details:
            usage = float(details.xpath("ns2:Consumption",namespaces=ns)[0].text)
            start = parse(details.xpath("ns2:StartDateTime",namespaces=ns)[0].text)
            end = parse(details.xpath("ns2:EndDateTime",namespaces=ns)[0].text)
            reading_type = details.xpath("ns2:ReadingType",namespaces=ns)
            if reading_type == []:
                estimated = False
            else:
                estimated = (reading_type[0].text == "estimated")
            records.append({"start": start, "end": end, "estimated": estimated, "value": usage})
        consumption_data = ConsumptionData(records, fuel_type, unit_str,
                record_type="arbitrary")
        consumption.append(consumption_data)

    return consumption

def import_green_button_xml(filename):
    """Import from Green Button XML.

    Parameters
    ----------
    filename : str
        Full path to Green Button XML file

    Returns
    -------
    out : eemeter.consumption.ConsumptionHistory
        Consumption history available for this project
    """
    with(open(filename,'r')) as f:
        tree = etree.parse(f)

    interval_blocks = tree.xpath("//*[local-name() = 'IntervalBlock']")

    def get_record(reading):
        usage_Wh = int(reading.xpath("*[local-name() = 'value']")[0].text)
        usage_kWh = usage_Wh / 1000.
        time_period = reading.xpath("*[local-name() = 'timePeriod']")[0]
        start_s = int(time_period.xpath("*[local-name() = 'start']")[0].text)
        start = datetime.fromtimestamp(start_s)
        duration_s = int(time_period.xpath("*[local-name() = 'duration']")[0].text)
        end_s = start_s + duration_s
        end = datetime.fromtimestamp(end_s)
        return {"value": usage_kWh, "start": start, "end": end}

    records = []
    for block in interval_blocks:
        readings = block.xpath("*[local-name() = 'IntervalReading']")
        records.extend([get_record(reading) for reading in readings])

    fuel_type = "electricity"
    unit_str = "kWh"
    record_type = "arbitrary"
    consumption_data = ConsumptionData(records, fuel_type, unit_str,
            record_type)
    return consumption_data

def import_seed_timeseries(db_url):
    """Import from SEED database

    Parameters
    ----------
    db_url : str
        SEED database url (should be interpretable by SQLAlchemy)

    Returns
    -------
    out : dict
        Dictionary of ConsumptionHistory objects keyed on BuildingSnapshot id
    """
    ENERGY_TYPES = {
        1: "natural_gas",
        2: "electricity",
        3: "fuel_oil",
        4: "fuel_oil_no_1",
        5: "fuel_oil_no_2",
        6: "fuel_oil_no_4",
        7: "fuel_oil_no_5_and_no_6",
        8: "district_steam",
        9: "district_hot_water",
        10: "district_chilled_water",
        11: "propane",
        12: "liquid_propane",
        13: "kerosene",
        14: "diesel",
        15: "coal",
        16: "coal_anthracite",
        17: "coal_bituminous",
        18: "coke",
        19: "wood",
        20: "other",
        21: "water",
    }

    ENERGY_UNITS = {
        1: "kWh",
        2: "therm",
        3: "Wh",
    }

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

    engine = create_engine(db_url)
    conn = engine.connect()
    s = select([seed_meter_building_snapshot])

    building_meters = {}
    for row in conn.execute(s):
        building_id = row["buildingsnapshot_id"]
        meter_id = row["meter_id"]
        if building_id not in building_meters:
            building_meters[building_id] = [meter_id]
        else:
            building_meters[building_id].append(meter_id)
    buildings_data = {}
    for building_id, meter_ids in building_meters.items():
        consumption = []
        for meter_id in meter_ids:
            meters = select([seed_meter]).where(seed_meter.c.id == meter_id)
            meter_row = conn.execute(meters).fetchone()
            fuel_type = ENERGY_TYPES[meter_row["energy_type"]]
            unit_name = ENERGY_UNITS[meter_row["energy_units"]]

            timeseries = select([seed_timeseries]).where(seed_timeseries.c.meter_id == meter_id)

            records = []
            for row in conn.execute(timeseries):
                start = row["begin_time"]
                end = row["end_time"]
                usage = row["reading"]
                record = {"start": start, "end": end, "value": usage}
                records.append(record)
            consumption_data = ConsumptionData(records, fuel_type, unit_name, record_type="arbitrary")
            consumption.append(consumption_data)

        buildings_data[building_id] = consumption

    return buildings_data

def import_pandas(df):
    """Import from pandas dataframe with the following columns:

    - Consumption: float
    - UnitofMeasure: {"therms", "kWh"}
    - FuelType: {"natural gas", "electricity"}
    - StartDateTime: str (ISO 8601 combined date time)
    - EndDateTime: str (ISO 8601 combined date time)
    - ReadingType: {"actual", "estimated"}

    Parameters
    ----------
    df : pandas.DataFrame
        pandas DataFrame with the Columns outlined above.

    Returns
    -------
    out : eemeter.consumption.ConsumptionData
        Consumption history for all consumptions stored in this DataFrame
    """
    fuel_type_mapping = {
        "electricity": "electricity",
        "natural gas": "natural_gas",
    }

    unit_of_measure_mapping = {
        "kWh": "kWh",
        "therms": "therm",
    }

    consumptions = []
    fuel_type = None
    unit_name = None
    records = []
    for i,row in df.iterrows():
        usage = row["Consumption"]

        # unit_name must be consistent
        _unit_name = unit_of_measure_mapping[row["UnitofMeasure"]]
        if unit_name is None:
            unit_name = _unit_name
        elif unit_name != _unit_name:
            message = "Inconsistent UnitofMeasure did not match {}: {}"\
                    .format(unit_name, _unit_name)
            warn(message)
            continue

        # fuel_type must be consistent
        _fuel_type = fuel_type_mapping[row["FuelType"]]
        if fuel_type is None:
            fuel_type = _fuel_type
        elif fuel_type != _fuel_type:
            message = "Inconsistent FuelType did not match {}: {}"\
                    .format(fuel_type, _fuel_type)
            warn(message)
            continue

        start_date = row["StartDateTime"]
        end_date = row["EndDateTime"]
        estimated = (row["ReadingType"] == "estimated")
        record = {"start": start_date, "end": end_date, "value": usage,
                "estimated": estimated}
        records.append(record)
    consumption_data = ConsumptionData(records, fuel_type, unit_name,
            record_type="arbitrary")
    return consumption_data

def import_csv(filename):
    """Import from csv spreadsheet with the following columns:

    - Consumption: float
    - UnitofMeasure: {"therms", "kWh"}
    - FuelType: {"natural gas", "electricity"}
    - StartDateTime: str (ISO 8601 combined date time)
    - EndDateTime: str (ISO 8601 combined date time)
    - ReadingType: {"actual", "estimated"}

    Parameters
    ----------
    filename : str
        Full path to CSV file

    Returns
    -------
    out : eemeter.consumption.ConsumptionHistory
        Consumption history stored in this file
    """
    df = pd.read_csv(filename)
    df['StartDateTime'] = pd.to_datetime(df['StartDateTime'])
    df['EndDateTime'] = pd.to_datetime(df['EndDateTime'])
    return import_pandas(df)

def import_excel(filename):
    """Import from excel spreadsheet with the following columns:

    - Consumption: float
    - UnitofMeasure: {"therms", "kWh"}
    - FuelType: {"natural gas", "electricity"}
    - StartDateTime: str (ISO 8601 combined date time)
    - EndDateTime: str (ISO 8601 combined date time)
    - ReadingType: {"actual", "estimated"}

    Parameters
    ----------
    filename : str
        Full path to XLSX file

    Returns
    -------
    out : eemeter.consumption.ConsumptionHistory
        Consumption history stored in this file
    """
    df = pd.read_excel(filename)
    df['StartDateTime'] = pd.to_datetime(df['StartDateTime'])
    df['EndDateTime'] = pd.to_datetime(df['EndDateTime'])
    return import_pandas(df)
