import logging
from collections import OrderedDict

import numpy as np
import pandas as pd
import pytz

from eemeter.processors.dispatchers import (
    get_approximate_frequency,
)

from eemeter.processors.location import (
    get_co2_source,
)

from eemeter.modeling.models import HourlyLoadProfileModel
from scipy import interpolate

logger = logging.getLogger(__name__)


def unpack(modeled_trace, baseline_label, reporting_label,
           baseline_period, reporting_period,
           weather_source, weather_normal_source,
           site, derivative_freq='D'):

    baseline_output = modeled_trace.fit_outputs[baseline_label]
    reporting_output = modeled_trace.fit_outputs[reporting_label]

    baseline_model_success = (baseline_output["status"] == "SUCCESS")
    reporting_model_success = (reporting_output["status"] == "SUCCESS")

    formatter = modeled_trace.formatter
    trace = modeled_trace.trace

    co2_source = get_co2_source(site)

    # default project dates
    baseline_start_date = baseline_period.start_date
    baseline_end_date = baseline_period.end_date
    reporting_start_date = reporting_period.start_date
    reporting_end_date = reporting_period.end_date

    # Note: observed data uses project dates, not data dates
    # convert trace data to daily
    if derivative_freq == 'H':
        trace_data = formatter.hourly_trace_data(trace)
        normalyear_periods = 365*24
    else:
        derivative_freq = 'D'
        trace_data = formatter.daily_trace_data(trace)
        normalyear_periods = 365
    if trace_data.empty:
        return None

    trace_frequency = get_approximate_frequency(trace)
    if trace_frequency in ['H', '15T', '30T']:
        hourly_trace_data = formatter.hourly_trace_data(trace)
    else:
        hourly_trace_data = None

    if baseline_start_date is None:
        baseline_period_data = \
            trace_data[:baseline_end_date].copy()
        if hourly_trace_data is not None:
            hourly_baseline_period_data = \
                hourly_trace_data[:baseline_end_date].copy()
        else:
            hourly_baseline_period_data = None
    else:
        baseline_period_data = \
            trace_data[baseline_start_date:baseline_end_date].copy()
        if hourly_trace_data is not None:
            hourly_baseline_period_data = \
                hourly_trace_data[baseline_start_date:baseline_end_date].copy()
        else:
            hourly_baseline_period_data = None

    project_period_data = \
        trace_data[baseline_end_date:reporting_start_date].copy()

    if reporting_end_date is None:
        reporting_period_data = \
            trace_data[reporting_start_date:].copy()
        if hourly_trace_data is not None:
            hourly_reporting_period_data = \
                hourly_trace_data[reporting_start_date:].copy()
        else:
            hourly_reporting_period_data = None
    else:
        reporting_period_data = \
            trace_data[reporting_start_date:reporting_end_date].copy()
        if hourly_trace_data is not None:
            hourly_reporting_period_data = \
                hourly_trace_data[reporting_start_date:reporting_end_date].copy()
        else:
            hourly_reporting_period_data = None

    weather_source_success = (weather_source is not None)
    weather_normal_source_success = (weather_normal_source is not None)

    # annualized fixture
    if weather_normal_source_success:
        normal_index = pd.date_range(
            '2015-01-01', freq=derivative_freq, periods=normalyear_periods,
            tz=pytz.UTC)
        annualized_fixture = formatter.create_demand_fixture(
            normal_index, weather_normal_source)
        if hourly_trace_data is not None:
            normal_index = pd.date_range(
                '2015-01-01', freq='H', periods=365*24,
                tz=pytz.UTC)
            hourly_annualized_fixture = formatter.create_demand_fixture(
                normal_index, weather_normal_source)
        else:
            hourly_annualized_fixture = None
    else:
        annualized_fixture = None
        hourly_annualized_fixture = None

    # find start and end dates of reporting data
    if not reporting_period_data.empty:
        reporting_data_start_date = reporting_period_data.index[0]
        reporting_data_end_date = reporting_period_data.index[-1]
    else:
        reporting_data_start_date = reporting_start_date
        reporting_data_end_date = reporting_start_date

    if not baseline_period_data.empty:
        baseline_data_start_date = baseline_period_data.index[0]
        baseline_data_end_date = baseline_period_data.index[-1]
    else:
        baseline_data_start_date = baseline_end_date
        baseline_data_end_date = baseline_end_date

    baseline_model = modeled_trace.model_mapping[baseline_label]
    reporting_model = modeled_trace.model_mapping[reporting_label]

    # reporting period fixture
    if None not in (
        reporting_data_start_date, reporting_data_end_date) and \
            weather_source_success:

        if reporting_data_start_date == reporting_data_end_date:
            reporting_period_index = pd.Series([])
        else:
            reporting_period_index = pd.date_range(
                start=reporting_data_start_date,
                end=reporting_data_end_date,
                freq=derivative_freq,
                tz=pytz.UTC)

        reporting_period_fixture = formatter.create_demand_fixture(
            reporting_period_index, weather_source)
        reporting_period_fixture_success = True
        if len(reporting_period_fixture) == 0:
            reporting_period_fixture_success = False
        if hourly_trace_data is not None:
            reporting_period_index = pd.date_range(
                start=reporting_data_start_date,
                end=reporting_data_end_date,
                freq='H',
                tz=pytz.UTC)
            hourly_reporting_period_fixture = formatter.create_demand_fixture(
                reporting_period_index, weather_source)
        else:
            hourly_reporting_period_fixture = None

        # Apply mask which indicates where data is missing (with daily
        # resolution)
        unmasked_reporting_period_fixture = \
            reporting_period_fixture.copy()
        if 'input_mask' in reporting_output.keys():
            reporting_mask = reporting_output['input_mask']
            for i, mask in reporting_mask.iteritems():
                if pd.isnull(mask):
                    reporting_period_fixture[i] = np.nan
        else:
            reporting_mask = pd.Series([])
    else:
        reporting_mask = pd.Series([])
        reporting_period_fixture = None
        unmasked_reporting_period_fixture = None
        reporting_period_fixture_success = False
        hourly_reporting_period_fixture = None

    if None not in (
        baseline_data_start_date, baseline_data_end_date) and \
            weather_source_success:

        if baseline_data_start_date == baseline_data_end_date:
            baseline_period_index = pd.Series([])
        else:
            baseline_period_index = pd.date_range(
                start=baseline_data_start_date,
                end=baseline_data_end_date,
                freq=derivative_freq,
                tz=pytz.UTC)

        baseline_period_fixture = formatter.create_demand_fixture(
            baseline_period_index, weather_source)
        baseline_period_fixture_success = True
        if len(baseline_period_fixture) == 0:
            baseline_period_fixture_success = False
        if hourly_trace_data is not None:
            baseline_period_index = pd.date_range(
                start=baseline_data_start_date,
                end=baseline_data_end_date,
                freq='H',
                tz=pytz.UTC)
            hourly_baseline_period_fixture = formatter.create_demand_fixture(
                baseline_period_index, weather_source)
        else:
            hourly_baseline_period_fixture = None

        unmasked_baseline_period_fixture = \
            baseline_period_fixture.copy()
        if 'input_mask' in baseline_output.keys():
            baseline_mask = baseline_output['input_mask']
            for i, mask in baseline_mask.iteritems():
                if pd.isnull(mask):
                    baseline_period_fixture[i] = np.nan
        else:
            baseline_mask = pd.Series([])

    else:
        unmasked_baseline_period_fixture = None
        baseline_mask = pd.Series([])
        baseline_period_fixture = None
        baseline_period_fixture_success = False
        hourly_baseline_period_fixture = None
    return {
            'formatter': formatter,
            'trace': trace,
            'baseline_output': baseline_output,
            'reporting_output': reporting_output,
            'baseline_model_success': baseline_model_success,
            'reporting_model_success': reporting_model_success,
            'baseline_start_date': baseline_start_date,
            'baseline_end_date': baseline_end_date,
            'baseline_data_start_date': baseline_data_start_date,
            'baseline_data_end_date': baseline_data_end_date,
            'reporting_start_date': reporting_start_date,
            'reporting_end_date': reporting_end_date,
            'reporting_data_start_date': reporting_data_start_date,
            'reporting_data_end_date': reporting_data_end_date,
            'baseline_period_data': baseline_period_data,
            'project_period_data': project_period_data,
            'reporting_period_data': reporting_period_data,
            'weather_source_success': weather_source_success,
            'weather_normal_source_success': weather_normal_source_success,
            'annualized_fixture': annualized_fixture,
            'baseline_model': baseline_model,
            'reporting_model': reporting_model,
            'baseline_period_fixture': baseline_period_fixture,
            'baseline_period_fixture_success': baseline_period_fixture_success,
            'reporting_period_fixture': reporting_period_fixture,
            'reporting_period_fixture_success': reporting_period_fixture_success,
            'baseline_mask': baseline_mask,
            'reporting_mask': reporting_mask,
            'unmasked_baseline_period_fixture': unmasked_baseline_period_fixture,
            'unmasked_reporting_period_fixture': unmasked_reporting_period_fixture,
            'hourly_baseline_period_data': hourly_baseline_period_data,
            'hourly_reporting_period_data': hourly_reporting_period_data,
            'hourly_baseline_period_fixture': hourly_baseline_period_fixture,
            'hourly_reporting_period_fixture': hourly_reporting_period_fixture,
            'hourly_annualized_fixture': hourly_annualized_fixture,
            'co2_source': co2_source,
            }


def subtract_value_variance_tuple(tuple1, tuple2):
    (val1, var1), (val2, var2) = tuple1, tuple2
    try:
        assert val1 is not None
        assert val2 is not None
        assert var1 is not None
        assert var2 is not None
    except:
        return (None, None)
    return (val1 - val2, (var1**2 + var2**2)**0.5)


def serialize_observed(series):
    return OrderedDict([
        (start.isoformat(), value)
        for start, value in series.iteritems()
    ])


def _report_failed_derivative(series):
    logger.warning(
        'Failed computing derivative (series={})'
        .format(series)
    )


def hdd_balance_point_baseline(deriv_input):
    series = 'Heating degree day balance point, baseline period'
    description = '''Best-fit heating degree day balance point,
                     if any, for baseline model'''

    if not deriv_input['baseline_model_success']:
        _report_failed_derivative(series)
        return None

    if 'model_fit' in deriv_input['baseline_output'].keys() and \
       'model_params' in deriv_input['baseline_output']['model_fit'] and \
       'hdd_bp' in deriv_input['baseline_output']['model_fit']['model_params']:

        value = deriv_input['baseline_output']['model_fit']['model_params']['hdd_bp']

        return {
                'series': series,
                'description': description,
                'orderable': [None, ],
                'value': [value, ],
                'variance': [None, ]
        }
    else:
        _report_failed_derivative(series)
        return None


def hdd_coefficient_baseline(deriv_input):
    series = 'Best-fit heating coefficient, baseline period'
    description = '''Best-fit heating coefficient,
                     if any, for baseline model'''

    if not deriv_input['baseline_model_success']:
        _report_failed_derivative(series)
        return None

    if 'model_fit' in deriv_input['baseline_output'].keys() and \
       'model_params' in deriv_input['baseline_output']['model_fit'] and \
       'hdd_bp' in deriv_input['baseline_output']['model_fit']['model_params'] and \
       'coefficients' in deriv_input['baseline_output']['model_fit']['model_params'] and \
       'HDD_' + str(deriv_input['baseline_output']['model_fit']['model_params']['hdd_bp']) \
            in deriv_input['baseline_output']['model_fit']['model_params']['coefficients']:
        value = deriv_input['baseline_output']['model_fit']['model_params']['coefficients'][
            'HDD_' + str(deriv_input['baseline_output']['model_fit']['model_params']['hdd_bp'])]

        return {
                'series': series,
                'description': description,
                'orderable': [None, ],
                'value': [value, ],
                'variance': [None, ]
               }
    else:
        _report_failed_derivative(series)
        return None


def cdd_balance_point_baseline(deriv_input):
    series = 'Cooling degree day balance point, baseline period'
    description = '''Best-fit cooling degree day balance point,
                     if any, for baseline model'''

    if not deriv_input['baseline_model_success']:
        _report_failed_derivative(series)
        return None

    if 'model_fit' in deriv_input['baseline_output'].keys() and \
       'model_params' in deriv_input['baseline_output']['model_fit'] and \
       'cdd_bp' in deriv_input['baseline_output']['model_fit']['model_params']:
        value = deriv_input['baseline_output']['model_fit']['model_params']['cdd_bp']

        return {
                'series': series,
                'description': description,
                'orderable': [None, ],
                'value': [value, ],
                'variance': [None, ]
               }
    else:
        _report_failed_derivative(series)
        return None


def cdd_coefficient_baseline(deriv_input):
    series = 'Best-fit cooling coefficient, baseline period'
    description = '''Best-fit cooling coefficient,
                     if any, for baseline model'''

    if not deriv_input['baseline_model_success']:
        _report_failed_derivative(series)
        return None

    if 'model_fit' in deriv_input['baseline_output'].keys() and \
       'model_params' in deriv_input['baseline_output']['model_fit'] and \
       'cdd_bp' in deriv_input['baseline_output']['model_fit']['model_params'] and \
       'coefficients' in deriv_input['baseline_output']['model_fit']['model_params'] and \
       'CDD_' + str(deriv_input['baseline_output']['model_fit']['model_params']['cdd_bp']) \
            in deriv_input['baseline_output']['model_fit']['model_params']['coefficients']:
        value = deriv_input['baseline_output']['model_fit']['model_params']['coefficients'][
            'CDD_' + str(deriv_input['baseline_output']['model_fit']['model_params']['cdd_bp'])]

        return {
                'series': series,
                'description': description,
                'orderable': [None, ],
                'value': [value, ],
                'variance': [None, ]
               }
    else:
        _report_failed_derivative(series)
        return None


def intercept_baseline(deriv_input):
    series = 'Best-fit intercept, baseline period'
    description = '''Best-fit intercept, if any, for baseline model'''

    if not deriv_input['baseline_model_success']:
        _report_failed_derivative(series)
        return None

    if 'model_fit' in deriv_input['baseline_output'] and \
       'model_params' in deriv_input['baseline_output']['model_fit'] and \
       'coefficients' in deriv_input['baseline_output']['model_fit']['model_params'] and \
       'Intercept' in deriv_input['baseline_output']['model_fit']['model_params']['coefficients']:
        value = deriv_input['baseline_output']['model_fit']['model_params']['coefficients']['Intercept']

        return {
                'series': series,
                'description': description,
                'orderable': [None, ],
                'value': [value, ],
                'variance': [None, ]
               }
    else:
        _report_failed_derivative(series)
        return None


def hdd_balance_point_reporting(deriv_input):
    series = 'Heating degree day balance point, reporting period'
    description = '''Best-fit heating degree day balance point,
                     if any, for reporting model'''

    if not deriv_input['reporting_model_success']:
        _report_failed_derivative(series)
        return None

    if 'model_fit' in deriv_input['reporting_output'].keys() and \
       'model_params' in deriv_input['reporting_output']['model_fit'] and \
       'hdd_bp' in deriv_input['reporting_output']['model_fit']['model_params']:

        value = deriv_input['reporting_output']['model_fit']['model_params']['hdd_bp']

        return {
                'series': series,
                'description': description,
                'orderable': [None, ],
                'value': [value, ],
                'variance': [None, ]
        }
    else:
        _report_failed_derivative(series)
        return None


def hdd_coefficient_reporting(deriv_input):
    series = 'Best-fit heating coefficient, reporting period'
    description = '''Best-fit heating coefficient,
                     if any, for reporting model'''

    if not deriv_input['reporting_model_success']:
        _report_failed_derivative(series)
        return None

    if 'model_fit' in deriv_input['reporting_output'].keys() and \
       'model_params' in deriv_input['reporting_output']['model_fit'] and \
       'hdd_bp' in deriv_input['reporting_output']['model_fit']['model_params'] and \
       'coefficients' in deriv_input['reporting_output']['model_fit']['model_params'] and \
       'HDD_' + str(deriv_input['reporting_output']['model_fit']['model_params']['hdd_bp']) \
            in deriv_input['reporting_output']['model_fit']['model_params']['coefficients']:
        value = deriv_input['reporting_output']['model_fit']['model_params']['coefficients'][
            'HDD_' + str(deriv_input['reporting_output']['model_fit']['model_params']['hdd_bp'])]

        return {
                'series': series,
                'description': description,
                'orderable': [None, ],
                'value': [value, ],
                'variance': [None, ]
               }
    else:
        _report_failed_derivative(series)
        return None


def cdd_balance_point_reporting(deriv_input):
    series = 'Cooling degree day balance point, reporting period'
    description = '''Best-fit cooling degree day balance point,
                     if any, for reporting model'''

    if not deriv_input['reporting_model_success']:
        _report_failed_derivative(series)
        return None

    if 'model_fit' in deriv_input['reporting_output'].keys() and \
       'model_params' in deriv_input['reporting_output']['model_fit'] and \
       'cdd_bp' in deriv_input['reporting_output']['model_fit']['model_params']:
        value = deriv_input['reporting_output']['model_fit']['model_params']['cdd_bp']

        return {
                'series': series,
                'description': description,
                'orderable': [None, ],
                'value': [value, ],
                'variance': [None, ]
               }
    else:
        _report_failed_derivative(series)
        return None


def cdd_coefficient_reporting(deriv_input):
    series = 'Best-fit cooling coefficient, reporting period'
    description = '''Best-fit cooling coefficient,
                     if any, for reporting model'''

    if not deriv_input['reporting_model_success']:
        _report_failed_derivative(series)
        return None

    if 'model_fit' in deriv_input['reporting_output'].keys() and \
       'model_params' in deriv_input['reporting_output']['model_fit'] and \
       'cdd_bp' in deriv_input['reporting_output']['model_fit']['model_params'] and \
       'coefficients' in deriv_input['reporting_output']['model_fit']['model_params'] and \
       'CDD_' + str(deriv_input['reporting_output']['model_fit']['model_params']['cdd_bp']) \
            in deriv_input['reporting_output']['model_fit']['model_params']['coefficients']:
        value = deriv_input['reporting_output']['model_fit']['model_params']['coefficients'][
            'CDD_' + str(deriv_input['reporting_output']['model_fit']['model_params']['cdd_bp'])]

        return {
                'series': series,
                'description': description,
                'orderable': [None, ],
                'value': [value, ],
                'variance': [None, ]
               }
    else:
        _report_failed_derivative(series)
        return None


def intercept_reporting(deriv_input):
    series = 'Best-fit intercept, reporting period'
    description = '''Best-fit intercept, if any, for reporting model'''

    if not deriv_input['reporting_model_success']:
        _report_failed_derivative(series)
        return None

    if 'model_fit' in deriv_input['reporting_output'] and \
       'model_params' in deriv_input['reporting_output']['model_fit'] and \
       'coefficients' in deriv_input['reporting_output']['model_fit']['model_params'] and \
       'Intercept' in deriv_input['reporting_output']['model_fit']['model_params']['coefficients']:
        value = deriv_input['reporting_output']['model_fit']['model_params']['coefficients']['Intercept']

        return {
                'series': series,
                'description': description,
                'orderable': [None, ],
                'value': [value, ],
                'variance': [None, ]
               }
    else:
        _report_failed_derivative(series)
        return None


def cumulative_baseline_model_minus_reporting_model_normal_year(deriv_input):
    series = 'Cumulative baseline model minus reporting model, normal year'
    description = '''Total predicted usage according to the baseline model
                     over the normal weather year, minus the total predicted
                     usage according to the reporting model over the normal
                     weather year. Days for which normal year weather data
                     does not exist are removed.'''

    if (not deriv_input['baseline_model_success']) or \
       (not deriv_input['reporting_model_success']) or \
       (not deriv_input['weather_normal_source_success']):
        _report_failed_derivative(series)
        return None

    try:
        value, variance = subtract_value_variance_tuple(
            deriv_input['baseline_model'].predict(deriv_input['annualized_fixture'], summed=True),
            deriv_input['reporting_model'].predict(deriv_input['annualized_fixture'], summed=True))
        return {
                'series': series,
                'description': description,
                'orderable': [None, ],
                'value': [value, ],
                'variance': [variance, ]
               }
    except:
        _report_failed_derivative(series)
        return None


def baseline_model_minus_reporting_model_normal_year(deriv_input):
    series = 'Baseline model minus reporting model, normal year'
    description = '''Predicted usage according to the baseline model
                     over the normal weather year, minus the predicted
                     usage according to the reporting model over the normal
                     weather year.'''

    if (not deriv_input['baseline_model_success']) or \
       (not deriv_input['reporting_model_success']) or \
       (not deriv_input['weather_normal_source_success']):
        _report_failed_derivative(series)
        return None

    try:
        value, variance = subtract_value_variance_tuple(
            deriv_input['baseline_model'].predict(deriv_input['annualized_fixture'], summed=False),
            deriv_input['reporting_model'].predict(deriv_input['annualized_fixture'], summed=False))
        return {
                'series': series,
                'description': description,
                'orderable': [i.isoformat() for i in deriv_input['annualized_fixture'].index],
                'value': value.tolist(),
                'variance': variance.tolist()
               }
    except:
        _report_failed_derivative(series)
        return None


def cumulative_baseline_model_normal_year(deriv_input):
    series = 'Cumulative baseline model, normal year'
    description = '''Total predicted usage according to the baseline model
                     over the normal weather year. Days for which normal
                     year weather data does not exist are removed.'''

    if (not deriv_input['baseline_model_success']) or \
       (not deriv_input['weather_normal_source_success']):
        _report_failed_derivative(series)
        return None

    try:
        value, variance = deriv_input['baseline_model'].predict(
                          deriv_input['annualized_fixture'], summed=True)
        return {
                'series': series,
                'description': description,
                'orderable': [None, ],
                'value': [value, ],
                'variance': [variance, ]
               }
    except:
        _report_failed_derivative(series)
        return None


def baseline_model_normal_year(deriv_input):
    series = 'Baseline model, normal year'
    description = '''Predicted usage according to the baseline model
                     over the normal weather year.'''

    if (not deriv_input['baseline_model_success']) or \
       (not deriv_input['weather_normal_source_success']):
        _report_failed_derivative(series)
        return None

    try:
        value, variance = deriv_input['baseline_model'].predict(
                    deriv_input['annualized_fixture'], summed=False)
        return {
                'series': series,
                'description': description,
                'orderable': [i.isoformat() for i in deriv_input['annualized_fixture'].index],
                'value': value.tolist(),
                'variance': variance.tolist()
               }
    except:
        _report_failed_derivative(series)
        return None


def cumulative_baseline_model_reporting_period(deriv_input):
    series = 'Cumulative baseline model, reporting period'
    description = '''Total predicted usage according to the baseline model
                     over the reporting period. Days for which reporting
                     period weather data does not exist are removed.'''

    if (not deriv_input['weather_source_success']) or \
       (not deriv_input['reporting_period_fixture_success']) or \
       (not deriv_input['baseline_model_success']):
        _report_failed_derivative(series)
        return None

    try:
        value, variance = deriv_input['baseline_model'].predict(
                          deriv_input['reporting_period_fixture'], summed=True)
        return {
                'series': series,
                'description': description,
                'orderable': [None, ],
                'value': [value, ],
                'variance': [variance, ]
               }
    except:
        _report_failed_derivative(series)
        return None


def baseline_model_reporting_period(deriv_input):
    series = 'Baseline model, reporting period'
    description = '''Predicted usage according to the baseline model
                     over the reporting period.'''

    if (not deriv_input['weather_source_success']) or \
       (not deriv_input['reporting_period_fixture_success']) or \
       (not deriv_input['baseline_model_success']):
        _report_failed_derivative(series)
        return None

    try:
        value, variance = deriv_input['baseline_model'].predict(
                          deriv_input['reporting_period_fixture'], summed=False)
        return {
                'series': series,
                'description': description,
                'orderable': [i.isoformat() for i in deriv_input['reporting_period_fixture'].index],
                'value': value.tolist(),
                'variance': variance.tolist()
               }
    except:
        _report_failed_derivative(series)
        return None


def masked_baseline_model_reporting_period(deriv_input):
    series = 'Masked baseline model, reporting period'
    description = '''Predicted usage according to the baseline model
                     over the reporting period, null where values are
                     missing in either the observed usage or
                     temperature data.'''

    if (not deriv_input['weather_source_success']) or \
       (not deriv_input['reporting_period_fixture_success']) or \
       (not deriv_input['baseline_model_success']):
        _report_failed_derivative(series)
        return None

    try:
        value, variance = deriv_input['baseline_model'].predict(
                          deriv_input['reporting_period_fixture'], summed=False)
        return {
                'series': series,
                'description': description,
                'orderable': [i.isoformat() for i in deriv_input['reporting_period_fixture'].index],
                'value': [
                    v if not deriv_input['reporting_mask'].get(i, True) else None
                    for i, v in value.iteritems()
                ],
                'variance': [
                    v if not deriv_input['reporting_mask'].get(i, True) else None
                    for i, v in variance.iteritems()
                ]
               }
    except:
        _report_failed_derivative(series)
        return None


def cumulative_baseline_model_minus_observed_reporting_period(deriv_input):
    series = 'Cumulative baseline model minus observed, reporting period'
    description = '''Total predicted usage according to the baseline model
                     minus observed usage over the reporting period.
                     Days for which reporting period weather data or usage
                     do not exist are removed.'''

    if (not deriv_input['weather_source_success']) or \
       (not deriv_input['reporting_period_fixture_success']) or \
       (not deriv_input['baseline_model_success']):
        _report_failed_derivative(series)
        return None

    try:
        value, variance = subtract_value_variance_tuple(
                deriv_input['baseline_model'].predict(
                    deriv_input['reporting_period_fixture'], summed=True),
                (deriv_input['reporting_period_data'].sum(), 0)
            )
        return {
                'series': series,
                'description': description,
                'orderable': [None, ],
                'value': [value, ],
                'variance': [variance, ]
               }
    except:
        _report_failed_derivative(series)
        return None


def baseline_model_minus_observed_reporting_period(deriv_input):
    series = 'Baseline model minus observed, reporting period'
    description = '''Predicted usage according to the baseline model
                     minus observed usage over the reporting period.'''

    if (not deriv_input['weather_source_success']) or \
       (not deriv_input['reporting_period_fixture_success']) or \
       (not deriv_input['baseline_model_success']):
        _report_failed_derivative(series)
        return None
    try:
        value, variance = subtract_value_variance_tuple(
                deriv_input['baseline_model'].predict(
                    deriv_input['reporting_period_fixture'], summed=False),
                (deriv_input['reporting_period_data'], 0)
            )

        return {
                'series': series,
                'description': description,
                'orderable': [i.isoformat() for i in deriv_input['reporting_period_fixture'].index],
                'value': value.tolist(),
                'variance': variance.tolist()
               }
    except:
        _report_failed_derivative(series)
        return None


def masked_baseline_model_minus_observed_reporting_period(deriv_input):
    series = 'Masked baseline model minus observed, reporting period'
    description = '''Predicted usage according to the baseline model
                     minus observed usage over the reporting period,
                     null where values are missing in either
                     the observed usage or temperature data.'''

    if (not deriv_input['weather_source_success']) or \
       (not deriv_input['reporting_period_fixture_success']) or \
       (not deriv_input['baseline_model_success']):
        _report_failed_derivative(series)
        return None
    try:
        value, variance = subtract_value_variance_tuple(
                deriv_input['baseline_model'].predict(
                    deriv_input['reporting_period_fixture'], summed=False),
                (deriv_input['reporting_period_data'], 0)
            )
        return {
                'series': series,
                'description': description,
                'orderable': [i.isoformat() for i in deriv_input['reporting_period_fixture'].index],
                'value': [
                    v if not deriv_input['reporting_mask'].get(i, True) else None
                    for i, v in value.iteritems()
                 ],
                'variance': [
                    v if not deriv_input['reporting_mask'].get(i, True) else None
                    for i, v in variance.iteritems()
                 ]
                }
    except:
        _report_failed_derivative(series)
        return None


def baseline_model_baseline_period(deriv_input):
    series = 'Baseline model, baseline period'
    description = '''Predicted usage according to the baseline model
                     over the baseline period.'''

    if (not deriv_input['baseline_period_fixture_success']) or \
       (not deriv_input['baseline_model_success']):
        _report_failed_derivative(series)
        return None
    try:
        value, variance = deriv_input['baseline_model'].predict(
                          deriv_input['baseline_period_fixture'], summed=False)
        return {
                'series': series,
                'description': description,
                'orderable': [i.isoformat() for i in deriv_input['baseline_period_fixture'].index],
                'value': value.tolist(),
                'variance': variance.tolist()
               }
    except:
        _report_failed_derivative(series)
        return None


def cumulative_reporting_model_normal_year(deriv_input):
    series = 'Cumulative reporting model, normal year'
    description = '''Total predicted usage according to the reporting model
                     over the reporting period.  Days for which normal year
                     weather data does not exist are removed.'''

    if (not deriv_input['weather_normal_source_success']) or \
       (not deriv_input['reporting_model_success']):
        _report_failed_derivative(series)
        return None

    try:
        value, variance = deriv_input['reporting_model'].predict(
                    deriv_input['annualized_fixture'], summed=True)
        return {
                'series': series,
                'description': description,
                'orderable': [None, ],
                'value': [value, ],
                'variance': [variance, ]
               }
    except:
        _report_failed_derivative(series)
        return None


def reporting_model_normal_year(deriv_input):
    series = 'Reporting model, normal year'
    description = '''Predicted usage according to the reporting model
                     over the reporting period.'''

    if (not deriv_input['weather_normal_source_success']) or \
       (not deriv_input['reporting_model_success']):
        _report_failed_derivative(series)
        return None

    try:
        value, variance = deriv_input['reporting_model'].predict(
                    deriv_input['annualized_fixture'], summed=False)
        return {
                'series': series,
                'description': description,
                'orderable': [i.isoformat() for i in deriv_input['annualized_fixture'].index],
                'value': value.tolist(),
                'variance': variance.tolist()
               }
    except:
        _report_failed_derivative(series)
        return None


def reporting_model_reporting_period(deriv_input):
    series = 'Reporting model, reporting period'
    description = '''Predicted usage according to the reporting model
                     over the reporting period.'''

    if (not deriv_input['reporting_period_fixture_success']) or \
       (not deriv_input['reporting_model_success']):
        _report_failed_derivative(series)
        return None

    try:
        value, variance = deriv_input['reporting_model'].predict(
                deriv_input['reporting_period_fixture'], summed=False)
        return {
                'series': series,
                'description': description,
                'orderable': [i.isoformat() for i in deriv_input['reporting_period_fixture'].index],
                'value': value.tolist(),
                'variance': variance.tolist()
               }
    except:
        _report_failed_derivative(series)
        return None


def cumulative_observed_reporting_period(deriv_input):
    series = 'Cumulative observed, reporting period'
    description = '''Total observed usage over the reporting period.
                     Days for which weather data does not exist
                     are NOT removed.'''

    try:
        return {
                'series': series,
                'description': description,
                'orderable': [None],
                'value': [deriv_input['reporting_period_data'].sum(), ],
                'variance': [0, ]
               }
    except:
        _report_failed_derivative(series)
        return None


def observed_reporting_period(deriv_input):
    series = 'Observed, reporting period'
    description = '''Observed usage over the reporting period.'''
    try:
        return {
                'series': series,
                'description': description,
                'orderable': [i.isoformat() for i in deriv_input['reporting_period_data'].index],
                'value': deriv_input['reporting_period_data'].values.tolist(),
                'variance': [0 for _ in range(deriv_input['reporting_period_data'].shape[0])]
               }
    except:
        _report_failed_derivative(series)
        return None


def masked_observed_reporting_period(deriv_input):
    series = 'Masked observed, reporting period'
    description = '''Observed usage over the reporting period,
                     null where values are missing in either
                     the observed usage or temperature data.'''
    try:
        return {
                'series': series,
                'description': description,
                'orderable': [i.isoformat() for i in deriv_input['reporting_period_data'].index],
                'value': [
                    v if not deriv_input['reporting_mask'].get(i, True) else None
                    for i, v in deriv_input['reporting_period_data'].iteritems()
                ],
                'variance': [
                    0 if not deriv_input['reporting_mask'].get(i, True) else None
                    for i, v in deriv_input['reporting_period_data'].iteritems()
                ]
               }
    except:
        _report_failed_derivative(series)
        return None


def cumulative_observed_baseline_period(deriv_input):
    series = 'Cumulative observed, baseline period'
    description = '''Total observed usage over the baseline period.
                     Days for which weather data does not exist
                     are NOT removed.'''

    try:
        return {
                'series': series,
                'description': description,
                'orderable': [None],
                'value': [deriv_input['baseline_period_data'].sum(), ],
                'variance': [0, ]
               }
    except:
        _report_failed_derivative(series)
        return None


def observed_baseline_period(deriv_input):
    series = 'Observed, baseline period'
    description = '''Observed usage over the baseline period.'''
    try:
        return {
                'series': series,
                'description': description,
                'orderable': [i.isoformat() for i in deriv_input['baseline_period_data'].index],
                'value': deriv_input['baseline_period_data'].values.tolist(),
                'variance': [0 for _ in range(deriv_input['baseline_period_data'].shape[0])]
               }
    except:
        _report_failed_derivative(series)
        return None


def observed_project_period(deriv_input):
    series = 'Observed, project period'
    description = '''Observed usage over the project period.'''
    try:
        return {
                'series': series,
                'description': description,
                'orderable': [i.isoformat() for i in deriv_input['project_period_data'].index],
                'value': deriv_input['project_period_data'].values.tolist(),
                'variance': [0 for _ in range(deriv_input['project_period_data'].shape[0])]
               }
    except:
        _report_failed_derivative(series)
        return None


def temperature_baseline_period(deriv_input):
    series = 'Temperature, baseline period'
    description = '''Observed temperature (degF) over the baseline period.'''

    if (not deriv_input['weather_source_success']):
        _report_failed_derivative(series)
        return None

    try:
        return {
                'series': series,
                'description': description,
                'orderable': [
                    i.isoformat() for i in deriv_input['unmasked_baseline_period_fixture'].index],
                'value': deriv_input['unmasked_baseline_period_fixture']['tempF'].values.tolist(),
                'variance': [0 for _ in range(deriv_input['unmasked_baseline_period_fixture']['tempF'].shape[0])]
               }
    except:
        _report_failed_derivative(series)
        return None


def temperature_reporting_period(deriv_input):
    series = 'Temperature, reporting period'
    description = '''Observed temperature (degF) over the reporting period.'''

    if (not deriv_input['weather_source_success']):
        _report_failed_derivative(series)
        return None

    try:
        return {
                'series': series,
                'description': description,
                'orderable': [
                    i.isoformat() for i in deriv_input['unmasked_reporting_period_fixture'].index],
                'value': deriv_input['unmasked_reporting_period_fixture']['tempF'].values.tolist(),
                'variance': [0 for _ in range(deriv_input['unmasked_reporting_period_fixture']['tempF'].shape[0])]
               }
    except:
        _report_failed_derivative(series)
        return None


def masked_temperature_reporting_period(deriv_input):
    series = 'Masked temperature, reporting period'
    description = '''Observed temperature (degF) over the reporting
                     period, null where values are missing in either
                     the observed usage or temperature data.'''

    if (not deriv_input['weather_source_success']):
        _report_failed_derivative(series)
        return None
    try:
        return {
                'series': series,
                'description': description,
                'orderable': [
                    i.isoformat() for i in deriv_input['unmasked_reporting_period_fixture'].index],
                'value': [
                    v if not deriv_input['reporting_mask'].get(i, True) else None
                    for i, v in deriv_input['unmasked_reporting_period_fixture']['tempF'].iteritems()
                ],
                'variance': [
                    0 if not deriv_input['reporting_mask'].get(i, True) else None
                    for i, v in deriv_input['unmasked_reporting_period_fixture']['tempF'].iteritems()
                ]
               }
    except:
        _report_failed_derivative(series)
        return None


def temperature_normal_year(deriv_input):
    series = 'Temperature, normal year'
    description = '''Observed temperature (degF) over the normal year.'''

    if (not deriv_input['weather_normal_source_success']):
        _report_failed_derivative(series)
        return None
    try:
        return {
                'series': series,
                'description': description,
                'orderable': [
                    i.isoformat() for i in deriv_input['annualized_fixture'].index],
                'value': deriv_input['annualized_fixture']['tempF'].values.tolist(),
                'variance': [0 for _ in range(deriv_input['annualized_fixture']['tempF'].shape[0])]
               }
    except:
        _report_failed_derivative(series)
        return None


def baseline_mask(deriv_input):
    series = 'Inclusion mask, baseline period'
    description = '''Mask for baseline period data which is included in
                     model and savings cumulatives.'''
    try:
        return {
                'series': series,
                'description': description,
                'orderable': [i.isoformat() for i in deriv_input['baseline_mask'].index],
                'value': [bool(v) for v in deriv_input['baseline_mask'].values],
                'variance': [0 for _ in range(deriv_input['baseline_mask'].shape[0])]
               }
    except:
        _report_failed_derivative(series)
        return None


def reporting_mask(deriv_input):
    series = 'Inclusion mask, reporting period'
    description = '''Mask for reporting period data which is included in
                     model and savings cumulatives.'''
    try:
        return {
                'series': series,
                'description': description,
                'orderable': [i.isoformat() for i in deriv_input['reporting_mask'].index],
                'value': [bool(v) for v in deriv_input['reporting_mask'].values],
                'variance': [0 for _ in range(deriv_input['reporting_mask'].shape[0])]
               }
    except:
        _report_failed_derivative(series)
        return None


def normal_year_resource_curve(deriv_input):
    series = 'Resource curve, normal year'
    description = '''Hourly baseline load profile minus hourly reporting
                   load profile, normal year'''

    if (deriv_input['hourly_annualized_fixture'] is None) or \
       (deriv_input['hourly_baseline_period_data'] is None) or \
       (deriv_input['hourly_reporting_period_data'] is None) or \
       (not deriv_input['baseline_model_success']) or \
       (not deriv_input['reporting_model_success']):
        _report_failed_derivative(series)
        return None

    try:
        baseline_data_frame = pd.DataFrame(
            {'energy': deriv_input['hourly_baseline_period_data'].values},
            index=deriv_input['hourly_baseline_period_data'].index)
        baseline_load_profile_model = HourlyLoadProfileModel()
        baseline_load_profile_model.caltrack_model = deriv_input['baseline_model']
        baseline_load_profile_model.input_data = baseline_data_frame
        baseline_load_profile, baseline_var = baseline_load_profile_model.predict(
            deriv_input['hourly_annualized_fixture'], summed=False)

        reporting_data_frame = pd.DataFrame(
            {'energy': deriv_input['hourly_reporting_period_data'].values},
            index=deriv_input['hourly_reporting_period_data'].index)
        reporting_load_profile_model = HourlyLoadProfileModel()
        reporting_load_profile_model.caltrack_model = deriv_input['reporting_model']
        reporting_load_profile_model.input_data = reporting_data_frame
        reporting_load_profile, reporting_var = reporting_load_profile_model.predict(
            deriv_input['hourly_annualized_fixture'], summed=False)

        resource_curve = baseline_load_profile - reporting_load_profile
        resource_curve_var = reporting_var + baseline_var

        return {
                'series': series,
                'description': description,
                'orderable': [i.isoformat() for i in resource_curve.index],
                'value': [v for v in resource_curve.values],
                'variance': [v for v in resource_curve_var.values]
               }
    except:
        _report_failed_derivative(series)
        return None


def reporting_period_resource_curve(deriv_input):
    series = 'Resource curve, reporting period'
    description = '''Hourly baseline load profile minus hourly reporting
                   load profile, reporting period'''

    if (deriv_input['hourly_reporting_period_fixture'] is None) or \
       (deriv_input['hourly_baseline_period_data'] is None) or \
       (deriv_input['hourly_reporting_period_data'] is None) or \
       (not deriv_input['baseline_model_success']) or \
       (not deriv_input['reporting_model_success']):
        _report_failed_derivative(series)
        return None

    try:
        baseline_data_frame = pd.DataFrame(
            {'energy': deriv_input['hourly_baseline_period_data'].values},
            index=deriv_input['hourly_baseline_period_data'].index)
        baseline_load_profile_model = HourlyLoadProfileModel()
        baseline_load_profile_model.caltrack_model = deriv_input['baseline_model']
        baseline_load_profile_model.input_data = baseline_data_frame
        baseline_load_profile, baseline_var = baseline_load_profile_model.predict(
            deriv_input['hourly_reporting_period_fixture'], summed=False)

        reporting_data_frame = pd.DataFrame(
            {'energy': deriv_input['hourly_reporting_period_data'].values},
            index=deriv_input['hourly_reporting_period_data'].index)
        reporting_load_profile_model = HourlyLoadProfileModel()
        reporting_load_profile_model.caltrack_model = deriv_input['reporting_model']
        reporting_load_profile_model.input_data = reporting_data_frame
        reporting_load_profile, reporting_var = reporting_load_profile_model.predict(
            deriv_input['hourly_reporting_period_fixture'], summed=False)

        resource_curve = baseline_load_profile - reporting_load_profile
        resource_curve_var = reporting_var + baseline_var

        return {
                'series': series,
                'description': description,
                'orderable': [i.isoformat() for i in resource_curve.index],
                'value': [v for v in resource_curve.values],
                'variance': [v for v in resource_curve_var.values]
               }
    except:
        _report_failed_derivative(series)
        return None


def normal_year_co2_avoided(deriv_input, resource_curve):
    series = 'CO2 avoided emissions, normal year'
    description = '''Avoided CO2 emissions based on normal year
                   resource curve'''

    avert = deriv_input['co2_source']

    if (resource_curve is None) or (avert is None):
        _report_failed_derivative(series)
        return None

    try:
        co2_by_load = avert.get_co2_by_load()
        load_by_hour = avert.get_load_by_hour()
        load_by_hour = load_by_hour[~(
            (load_by_hour.index.day == 29) &
            (load_by_hour.index.month == 2))]

        # Calculate the pre-intervention CO2 emissions
        f = interpolate.interp1d(co2_by_load.index, co2_by_load.values)
        co2_pre = f(load_by_hour.values)

        # Calculate the post-internention load and CO2 emissions
        load_post = load_by_hour.values - resource_curve.values
        co2_post = f(load_post)

        # Return the savings
        avoided_emissions = pd.Series(co2_pre - co2_post,
                                      index=resource_curve.index)

        return {
                'series': series,
                'description': description,
                'orderable': [i.isoformat() for i in avoided_emissions.index],
                'value': [v for v in avoided_emissions.values],
                'variance': [None for v in avoided_emissions.values]
               }
    except:
        _report_failed_derivative(series)
        return None
