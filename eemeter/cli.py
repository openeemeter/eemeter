from collections import OrderedDict
import datetime
import pytz
import csv
import os
import click
import pandas as pd
from scipy import stats
import numpy as np
from eemeter.structures import EnergyTrace
from eemeter.io.serializers import ArbitraryStartSerializer
from eemeter.ee.meter import EnergyEfficiencyMeter


@click.group()
def cli():
    pass


def serialize_meter_input(
        trace, zipcode, retrofit_start_date, retrofit_end_date):
    data = OrderedDict([
        ("type", "SINGLE_TRACE_SIMPLE_PROJECT"),
        ("trace", trace_serializer(trace)),
        ("project", project_serializer(
            zipcode, retrofit_start_date, retrofit_end_date
        )),
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
        return datetime.datetime.strptime(raw, date_format)\
                                .replace(tzinfo=pytz.UTC)
    return reader


date_readers = [
    date_reader('%Y-%m-%d %H:%M:%S'),
    date_reader('%Y-%m-%dT%H:%M:%S'),
    date_reader('%Y-%m-%dT%H:%M:%SZ'),
]


def flexible_date_reader(raw):
    for reader in date_readers:
        try:
            return reader(raw)
        except:
            pass
    raise ValueError("Unable to parse date")


def build_trace(trace_records):
    if trace_records[0]['interpretation'] == 'gas':
        unit = "THM"
        interpretation = "NATURAL_GAS_CONSUMPTION_SUPPLIED"
    else:
        unit = "KWH"
        interpretation = "ELECTRICITY_CONSUMPTION_SUPPLIED"
    trace_object = EnergyTrace(
        records=trace_records,
        unit=unit,
        interpretation=interpretation,
        serializer=ArbitraryStartSerializer(),
        trace_id=trace_records[0]['project_id']
    )
    return trace_object


def build_traces(trace_records):
    current_trace_id = None
    current_trace = []
    trace_objects = []

    # Split the concatenated traces into individual traces
    for record in trace_records:
        trace_id = record["project_id"] + " " + record["interpretation"]
        if current_trace_id is None:
            current_trace_id = trace_id
        elif current_trace_id == trace_id:
            current_trace.append(record)
        else:
            trace_objects.append(build_trace(current_trace))
            current_trace = [record]
            current_trace_id = trace_id
    trace_objects.append(build_trace(current_trace))

    return trace_objects


def run_meter(project, trace_object):
    print("\n\nRunning a meter for %s %s" % (
        trace_object.trace_id, trace_object.interpretation)
    )
    meter_input = serialize_meter_input(
        trace_object,
        project['zipcode'],
        project['project_start'],
        project['project_end']
    )
    ee = EnergyEfficiencyMeter()
    meter_output = ee.evaluate(meter_input)

    # Compute and output the annualized weather normal
    series_name = \
        'Cumulative baseline model minus reporting model, normal year'
    awn = [i['value'][0] for i in meter_output['derivatives']
           if i['series'] == series_name][0]
    awn_var = [i['variance'][0] for i in meter_output['derivatives']
               if i['series'] == series_name][0]
    awn_confint = stats.norm.interval(0.68, loc=awn, scale=np.sqrt(awn_var))
    print("Normal year savings estimate:")
    print("  {:f}\n  68% confidence interval: ({:f}, {:f})".
          format(awn, awn_confint[0], awn_confint[1]))

    # Compute and output the weather normalized reporting period savings
    series_name = \
        'Cumulative baseline model minus observed, reporting period'
    rep = [i['value'][0] for i in meter_output['derivatives']
           if i['series'] == series_name][0]
    rep_var = [i['variance'][0] for i in meter_output['derivatives']
               if i['series'] == series_name][0]
    rep_confint = stats.norm.interval(0.68, loc=rep, scale=np.sqrt(rep_var))
    print("Reporting period savings estimate:")
    print("  {:f}\n  68% confidence interval: ({:f}, {:f})".
          format(rep, rep_confint[0], rep_confint[1]))


def _analyze(inputs_path):
    projects = read_csv(os.path.join(inputs_path, 'projects.csv'))
    traces = read_csv(os.path.join(inputs_path, 'traces.csv'))

    for row in traces:
        row['start'] = flexible_date_reader(row['start'])

    for row in projects:
        row['project_start'] = flexible_date_reader(row['project_start'])
        row['project_end'] = flexible_date_reader(row['project_end'])

    trace_objects = build_traces(traces)

    for project in projects:
        for trace_object in trace_objects:
            if trace_object.trace_id == project['project_id']:
                run_meter(project, trace_object)


@cli.command()
def sample():
    path = os.path.realpath(__file__)
    cwd = os.path.dirname(path)
    sample_inputs_path = os.path.join(cwd, 'sample_data')
    print("Going to analyze the sample data set")
    print("The latest documentation of the sample data can be found at:")
    print("<URL for sample data documentation>")
    _analyze(sample_inputs_path)


@cli.command()
@click.argument('inputs_path', type=click.Path(exists=True))
def analyze(inputs_path):
    _analyze(inputs_path)
