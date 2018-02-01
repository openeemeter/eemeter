from collections import OrderedDict
import csv
import datetime
import errno
import json
import logging
import os

import click
import pytz
import pandas as pd
from scipy import stats
import numpy as np

from eemeter.structures import EnergyTrace
from eemeter.io.serializers import ArbitraryStartSerializer
from eemeter.ee.meter import EnergyEfficiencyMeter
from eemeter.processors.dispatchers import (
    get_approximate_frequency,
)
from eemeter.modeling.models.caltrack import CaltrackMonthlyModel


logging.basicConfig()


@click.group()
def cli():
    '''
       \b
       Example usage:
           eemeter analyze /path/to/input/data
       Or:
           eemeter sample

       The latter will use the example dataset included in the Python package
       in the subdirectory "eemeter/sample_data".

       This command will analyze a set of "traces," i.e. time series
       of energy usage, using the eemeter. The input directory should
       contain two files: projects.csv, and traces.csv.

       projects.csv specifies the project locations and the start and end
       dates of the interventions to be analyzed:

       \b
           project_id, zipcode, project_start, project_end
           ABC, 60640, 2015-12-31 00:00:00, 2016-01-01 00:00:00
           ...

       traces.csv contains the usage time series

       \b
           start, value, project_id, interpretation
           2015-01-01 00:00:00, 0.61683272591, ABC, gas
           ...

       Several common date/time formats are accepted. The project_id is a
       freeform string, and the interpretation should be "electricity" or
       "gas".

       The most commonly-used outputs will be displayed on the terminal;
       a more complete set of outputs may be requested by adding the
       argument "--full-output". By default, the full output (if requested)
       is placed in the directory "eemeter_output"; this may be overridden
       using the argument "--output-dir=/path/to/output".

       By default, the eemter will impose a 12-month requirement on both
       the pre- and post-intervention usage time series. To ignore this
       requirement, pass the option "--ignore-data-sufficiency".

    '''


def slugify(value):
    """
    Normalizes string, converts to lowercase, removes non-alpha characters,
    and converts spaces to underscores.
    """
    value = value.strip().lower()
    value = value.replace(',', '')
    value = value.replace(' ', '_')
    return value


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
    date_reader('%Y-%m-%d'),
    date_reader('%m/%d/%Y'),
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
            current_trace.append(record)
        elif current_trace_id == trace_id:
            current_trace.append(record)
        else:
            trace_objects.append(build_trace(current_trace))
            current_trace = [record]
            current_trace_id = trace_id
    trace_objects.append(build_trace(current_trace))

    return trace_objects


def full_output(meter_output, dirname, trace_id):
    cwd = os.getcwd()
    if not os.path.exists(dirname):
        try:
            os.mkdir(dirname)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    try:
        os.chdir(dirname)
    except:
        print("WARNING: Cannot go to output directory "+dirname)
        os.chdir(cwd)
        return
    try:
        os.mkdir(trace_id)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
    try:
        os.chdir(trace_id)
    except:
        print("WARNING: Cannot create trace output directory "+trace_id)
        os.chdir(cwd)
        return

    with open('output.json', 'w') as f:
        json.dump(meter_output, f)

    with open('log', 'w') as f:
        for line in meter_output['logs']:
            f.write(line + '\n')

    for derivative in meter_output['derivatives']:
        series_name = slugify(derivative['series'])
        with open(series_name, 'w') as f:
            fcsv = csv.writer(f)
            fcsv.writerow(['Orderable', 'Value', 'Variance'])
            for o, v, va in zip(derivative['orderable'],
                                derivative['value'],
                                derivative['variance']):
                fcsv.writerow([o, v, va])

    print("Created full output in "+os.getcwd())
    os.chdir(cwd)


def basic_output(meter_output):
    # Compute and output the annualized weather normal
    series_name = \
        'Cumulative baseline model minus reporting model, normal year'
    awn = [i['value'][0] for i in meter_output['derivatives']
           if i['series'] == series_name]
    if len(awn) > 0:
        awn = awn[0]
    else:
        awn = None
    awn_var = [i['variance'][0] for i in meter_output['derivatives']
               if i['series'] == series_name]
    if len(awn_var) > 0:
        awn_var = awn_var[0]
    else:
        awn_var = None
    awn_confint = []
    if awn is not None and awn_var is not None:
        awn_confint = stats.norm.interval(0.68, loc=awn,
                                          scale=np.sqrt(awn_var))

    if len(awn_confint) > 1:
        print("Normal year savings estimate:")
        print("  {:f}\n  68% confidence interval: ({:f}, {:f})".
              format(awn, awn_confint[0], awn_confint[1]))
    else:
        print("Normal year savings estimates not computed due to error:")
        bl_traceback = meter_output[
                'modeled_energy_trace']['fits']['baseline']['traceback']
        rp_traceback = meter_output[
                'modeled_energy_trace']['fits']['reporting']['traceback']
        if bl_traceback is not None:
            print(bl_traceback)
        if rp_traceback is not None:
            print(rp_traceback)

    # Compute and output the weather normalized reporting period savings
    series_name = \
        'Cumulative baseline model minus observed, reporting period'
    rep = [i['value'][0] for i in meter_output['derivatives']
           if i['series'] == series_name]
    if len(rep) > 0:
        rep = rep[0]
    else:
        rep = None
    rep_var = [i['variance'][0] for i in meter_output['derivatives']
               if i['series'] == series_name]
    if len(rep_var) > 0:
        rep_var = rep_var[0]
    else:
        rep_var = None
    rep_confint = []
    if rep is not None and rep_var is not None:
        rep_confint = stats.norm.interval(0.68, loc=rep,
                                          scale=np.sqrt(rep_var))
    else:
        rep_confint = []

    if len(rep_confint) > 1:
        print("Reporting period savings estimate:")
        print("  {:f}\n  68% confidence interval: ({:f}, {:f})".
              format(rep, rep_confint[0], rep_confint[1]))
    else:
        print("Reporting period savings estimates not computed due to error:")
        print(meter_output['modeled_energy_trace']['fits'][
                           'baseline']['traceback'])


def run_meter(project, trace_object, options=None):
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

    if options is not None and \
            'ignore_data_sufficiency' in options.keys() and \
            options['ignore_data_sufficiency'] is True:
        trace_frequency = get_approximate_frequency(trace_object)
        if trace_frequency not in ['H', 'D', '15T', '30T']:
            trace_frequency = None
        selector = (trace_object.interpretation, trace_frequency)
        model = ee._get_model(None, selector)

        model_class, model_kwargs = model

        if model_class == CaltrackMonthlyModel:
            model_kwargs['min_contiguous_baseline_months'] = 0
            model_kwargs['min_contiguous_reporting_months'] = 0
        else:
            model_kwargs['min_contiguous_months'] = 0

    meter_output = ee.evaluate(meter_input)
    basic_output(meter_output)

    if options is not None and \
       'full_output' in options.keys() and \
       options['full_output']:
        trace_output_dir = trace_object.trace_id + '.' + \
                           trace_object.interpretation
        full_output(meter_output, options['output_dir'], trace_output_dir)
    return meter_output


def _analyze(inputs_path, options=None):
    projects, trace_objects = _load_projects_and_traces(inputs_path)

    meter_output_list = list()
    for project in projects:
        for trace_object in trace_objects:
            if trace_object.trace_id == project['project_id']:
                meter_outputs = run_meter(
                    project,
                    trace_object,
                    options=options
                )
                meter_output_list.append(meter_outputs)

    return meter_output_list


def _load_projects_and_traces(inputs_path):
    projects = read_csv(os.path.join(inputs_path, 'projects.csv'))
    traces = read_csv(os.path.join(inputs_path, 'traces.csv'))

    for row in traces:
        row['start'] = flexible_date_reader(row['start'])

    for row in projects:
        row['project_start'] = flexible_date_reader(row['project_start'])
        row['project_end'] = flexible_date_reader(row['project_end'])

    trace_objects = build_traces(traces)

    return projects, trace_objects


def _get_sample_inputs_path():
    path = os.path.realpath(__file__)
    cwd = os.path.dirname(path)
    sample_inputs_path = os.path.join(cwd, 'sample_data')
    return sample_inputs_path


@cli.command()
@click.option('--full-output', is_flag=True, default=False,
              help='Create full eemeter output files')
@click.option('--output-dir', default='eemeter_output',
              help='Directory in which to put the full eemeter output.')
def sample(full_output, output_dir):
    sample_inputs_path = _get_sample_inputs_path()
    options = {'full_output': full_output, 'output_dir': output_dir}
    print("Going to analyze the sample data set")
    print("")
    _analyze(sample_inputs_path, options)


@cli.command()
@click.argument('inputs_path', type=click.Path(exists=True))
@click.option('--ignore-data-sufficiency', is_flag=True,
              help='Ignore the data sufficiency requirements.')
@click.option('--full-output', is_flag=True, default=False,
              help='Create full eemeter output files')
@click.option('--output-dir', default='eemeter_output',
              help='Directory in which to put the full eemeter output.')
def analyze(inputs_path, ignore_data_sufficiency, full_output, output_dir):
    options = {'ignore_data_sufficiency': ignore_data_sufficiency,
               'full_output': full_output, 'output_dir': output_dir}
    _analyze(inputs_path, options=options)
