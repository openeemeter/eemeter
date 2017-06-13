from collections import OrderedDict
import datetime
import pytz
import csv
import os
import click
import pandas as pd
import eemeter
from eemeter.structures import EnergyTrace
from eemeter.io.serializers import ArbitraryStartSerializer
from eemeter.ee.meter import EnergyEfficiencyMeter

@click.group()
def cli():
    pass

def serialize_meter_input(trace, zipcode, retrofit_start_date, retrofit_end_date):
    data = OrderedDict([
        ("type", "SINGLE_TRACE_SIMPLE_PROJECT"),
        ("trace", trace_serializer(trace)),
        ("project", project_serializer(zipcode, retrofit_start_date, retrofit_end_date)),
    ])
    return data

def trace_serializer(trace):
    data = OrderedDict([
        ("type", "ARBITRARY_START"),
        ("interpretation", trace.interpretation),
        ("unit", trace.unit),
        ("trace_id", trace.trace_id),
        ("interval", trace.interval),
        ("records", [
            OrderedDict([
                ("start", start.isoformat()),
                ("value", record.value if pd.notnull(record.value) else None),
                ("estimated", bool(record.estimated)),
            ])
            for start, record in trace.data.iterrows()
        ]),
    ])
    return data

def project_serializer(zipcode, retrofit_start_date, retrofit_end_date):
    data = OrderedDict([
        ("type", "PROJECT_WITH_SINGLE_MODELING_PERIOD_GROUP"),
        ("zipcode", zipcode),
        ("project_id", 'PROJECT_ID_ABC'),
        ("modeling_period_group", OrderedDict([
            ("baseline_period", OrderedDict([
                ("start", None),
                ("end", retrofit_start_date.isoformat()),
            ])),
            ("reporting_period", OrderedDict([
                ("start", retrofit_end_date.isoformat()),
                ("end", None),
            ]))
        ]))
    ])
    return data

def read_csv(path):
	result = []
	with open(path) as f:
		reader = csv.DictReader(f)
		for row in reader:
			result.append(row)
	return result

def date_reader(date_format):
    def reader(raw):
        if raw.strip() == '':
            return None
        return datetime.datetime.strptime(raw, date_format).replace(tzinfo=pytz.UTC)
    return reader

iso_8601_reader = date_reader('%Y-%m-%dT%H:%M:%S')

@cli.command()
@click.argument('inputs_path', type=click.Path(exists=True))
def analyze(inputs_path):

	projects = read_csv(os.path.join(inputs_path, 'projects.csv'))
	traces = read_csv(os.path.join(inputs_path, 'traces.csv'))

	for row in traces:
		row['start'] = iso_8601_reader(row['start'])

	for row in projects:
		row['project_start'] = iso_8601_reader(row['project_start'])
		row['project_end'] = iso_8601_reader(row['project_end'])

	energy_trace = EnergyTrace(
	    records=traces,
	    unit="KWH",
	    interpretation="ELECTRICITY_CONSUMPTION_SUPPLIED",
	    serializer=ArbitraryStartSerializer(),
	    trace_id='123',
	    interval='daily'
	)

	for project in projects:
		meter_input = serialize_meter_input(
			energy_trace, 
			project['zipcode'],
	    	project['project_start'],
	    	project['project_end']
	    )
		ee = EnergyEfficiencyMeter()
		meter_output = ee.evaluate(meter_input)

		print("Normal year savings estimate:")
		print([i['value'][0] for i in meter_output['derivatives'] 
		       if i['series']=='Cumulative baseline model minus reporting model, normal year'][0])


