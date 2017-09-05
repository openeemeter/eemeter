from collections import OrderedDict
import datetime
import pytz
import csv
import os
import click
import pandas as pd
from scipy import stats
from tabulate import tabulate
import numpy as np
from eemeter.structures import EnergyTrace
from eemeter.io.serializers import ArbitraryStartSerializer
from eemeter.ee.meter import EnergyEfficiencyMeter


@click.group(
        context_settings={
            'help_option_names':['-h', '--help']
        })
@click.option(
    '--tablefmt', '-t',
    type=click.Choice([
        'orgtbl', 'jira', 'rst', 'psql', 'mediawiki', 
        'html', 'latex', 'pipe'
    ]),
    help='Display the results in a table format'
)
@click.option(
    '--show-all', '-a',
    is_flag=True,
    default=False,
    help='Show all outputs'
)
@click.pass_context
def cli(ctx, tablefmt, show_all):
    """
    This command line tool uses properly-formatted projects & traces
    CSV files as inputs to generate the annualized weather normalized
    savings estimate and the realized savings estimate over the 
    reporting year by default.

    Many additional outputs can be reported by eemeter, including
    daily time series and best-fit model parameters. Use the
    --show-all option listed below to see additional outputs.

    For purposes of testing, a sample data set is provided in 
    eemeter/sample_data.
    Visit https://www.openee.io/docs/docs-product-manuals#sample-data
    for latest documentation on sample data.

    Use this tool for testing, validating code and running one-off
    analyses of archival data.
    """
    ctx.obj = {}
    ctx.obj['TABLEFMT'] = tablefmt
    ctx.obj['SHOWALL'] = show_all


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


def derivatives_model_stats(series_derivatives):
    n = series_derivatives['value'][0]
    var = series_derivatives['variance'][0]
    confint = (0, 0)
    if var:
        confint = stats.norm.interval(0.68, loc=n, scale=np.sqrt(var))
    
    return (n, confint[0], confint[1])


def series_model_stats(meter_output, series_name):
    n = [i['value'][0] for i in meter_output['derivatives']
           if i['series'] == series_name][0]
    var = [i['variance'][0] for i in meter_output['derivatives']
               if i['series'] == series_name][0]
    confint = (0, 0)
    if var:
        confint = stats.norm.interval(0.68, loc=n, scale=np.sqrt(var))
    
    return (n, confint[0], confint[1])


def print_meter_results(trace_object, meter_output):
    print("\n\nRunning a meter for %s %s" % (
        trace_object.trace_id, trace_object.interpretation)
    )

    # Compute and output the annualized weather normal
    awn, awn_conf_lb, awn_conf_ub = series_model_stats(
        meter_output,
        'Cumulative baseline model minus reporting model, normal year')
    print("Normal year savings estimate:")
    print("  {:f}\n  68% confidence interval: ({:f}, {:f})".
          format(awn, awn_conf_lb, awn_conf_ub))

    # Compute and output the weather normalized reporting period savings
    rep, rep_conf_lb, rep_conf_ub = series_model_stats(
        meter_output,
        'Cumulative baseline model minus observed, reporting period')
    print("Reporting period savings estimate:")
    print("  {:f}\n  68% confidence interval: ({:f}, {:f})".
          format(rep, rep_conf_lb, rep_conf_ub))


def add_model_stats(
        out_data, series_derivatives, alt_name):
    n, conf_lb, conf_ub = derivatives_model_stats(series_derivatives)

    if not n:
        return out_data

    out_data['Model'].append(alt_name)
    out_data['Savings_Estimate'].append('{:,.0f} kWh'.format(n))
    out_data['68%_Confidence_Interval'].append(
            '({:.0f},{:.0f})'.format(conf_lb, conf_ub))

    return out_data


def print_meter_results_table(
        trace_object, meter_output, tablefmt, series_to_alt=None):
    print("\n\n {} {}".format(
        trace_object.trace_id, trace_object.interpretation))

    out_data = { 
        "Model":[], 
        "Savings_Estimate": [],
        "68%_Confidence_Interval": []
    }
    for i in meter_output['derivatives']:
        series = i['series']
        if series_to_alt and series not in series_to_alt:
            continue

        out_data = add_model_stats(
            out_data,
            i,
            series if not series_to_alt else series_to_alt[series]
        )

    df = pd.DataFrame.from_dict(out_data)
    df = df[['Model', 'Savings_Estimate', '68%_Confidence_Interval']]
    df.set_index('Model', inplace=True)
    print(tabulate(df, headers='keys', tablefmt=tablefmt, stralign="left"))


def print_default_meter_results_table(
        trace_object, meter_output, tablefmt):
    series_to_alt = dict()
    series_to_alt[
        'Cumulative baseline model minus reporting model, normal year'
    ] = 'Normal year savings estimate'
    series_to_alt[
        'Cumulative baseline model minus observed, reporting period'
    ] = 'Reporting period savings estimate'

    print_meter_results_table(
        trace_object, meter_output, tablefmt, series_to_alt)


def run_meter(project, trace_object):
    meter_input = serialize_meter_input(
        trace_object,
        project['zipcode'],
        project['project_start'],
        project['project_end']
    )
    ee = EnergyEfficiencyMeter()
    meter_output = ee.evaluate(meter_input)

    return meter_output


def _analyze(inputs_path, show_all, tablefmt):
    projects = read_csv(os.path.join(inputs_path, 'projects.csv'))
    traces = read_csv(os.path.join(inputs_path, 'traces.csv'))

    for row in traces:
        row['start'] = flexible_date_reader(row['start'])

    for row in projects:
        row['project_start'] = flexible_date_reader(row['project_start'])
        row['project_end'] = flexible_date_reader(row['project_end'])

    trace_objects = build_traces(traces)
    meter_output_list = list()

    for project in projects:
        for trace_object in trace_objects:
            if trace_object.trace_id == project['project_id']:
                meter_output = run_meter(project, trace_object)
                meter_output_list.append(meter_output)
                if show_all:
                    print_meter_results_table(
                        trace_object, meter_output, tablefmt)
                elif tablefmt:
                    print_default_meter_results_table(
                        trace_object, meter_output, tablefmt)
                else:
                    print_meter_results(trace_object, meter_output)

    return meter_output_list


@cli.command()
@click.pass_obj
def sample(ctx):
    path = os.path.realpath(__file__)
    cwd = os.path.dirname(path)
    sample_inputs_path = os.path.join(cwd, 'sample_data')
    print("Going to analyze the sample data set")
    print("The latest documentation of the sample data can be found at:")
    print("https://www.openee.io/docs/docs-product-manuals#sample-data")
    _analyze(sample_inputs_path, ctx.get('SHOWALL'), ctx.get('TABLEFMT'))


@cli.command()
@click.pass_obj
@click.argument('inputs_path', type=click.Path(exists=True))
def analyze(ctx, inputs_path):
    _analyze(inputs_path, ctx.get('SHOWALL'), ctx.get('TABLEFMT'))
