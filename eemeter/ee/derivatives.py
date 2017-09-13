import logging
from collections import OrderedDict

import numpy as np
import pandas as pd
import pytz

logger = logging.getLogger(__name__)


def unpack(modeled_trace, baseline_label, reporting_label,
           baseline_period, reporting_period,
           weather_source, weather_normal_source):

    baseline_output = modeled_trace.fit_outputs[baseline_label]
    reporting_output = modeled_trace.fit_outputs[reporting_label]

    baseline_model_success = (baseline_output["status"] == "SUCCESS")
    reporting_model_success = (reporting_output["status"] == "SUCCESS")

    formatter = modeled_trace.formatter
    trace = modeled_trace.trace

    # default project dates
    baseline_start_date = baseline_period.start_date
    baseline_end_date = baseline_period.end_date
    reporting_start_date = reporting_period.start_date
    reporting_end_date = reporting_period.end_date

    # Note: observed data uses project dates, not data dates
    # convert trace data to daily
    daily_trace_data = formatter.daily_trace_data(trace)
    if daily_trace_data.empty:
        return None

    if baseline_start_date is None:
        baseline_period_data = \
            daily_trace_data[:baseline_end_date].copy()
    else:
        baseline_period_data = \
            daily_trace_data[baseline_start_date:baseline_end_date].copy()

    project_period_data = \
        daily_trace_data[baseline_end_date:reporting_start_date].copy()

    if reporting_end_date is None:
        reporting_period_data = \
            daily_trace_data[reporting_start_date:].copy()
    else:
        reporting_period_data = \
            daily_trace_data[reporting_start_date:reporting_end_date].copy()

    weather_source_success = (weather_source is not None)
    weather_normal_source_success = (weather_normal_source is not None)

    # annualized fixture
    if weather_normal_source_success:
        normal_index = pd.date_range(
            '2015-01-01', freq='D', periods=365, tz=pytz.UTC)
        annualized_daily_fixture = formatter.create_demand_fixture(
            normal_index, weather_normal_source)
    else:
        annualized_daily_fixture = None

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
            reporting_data_start_date, reporting_data_end_date) and weather_source_success:

        if reporting_data_start_date == reporting_data_end_date:
            reporting_period_daily_index = pd.Series([])
        else:
            reporting_period_daily_index = pd.date_range(
                start=reporting_data_start_date,
                end=reporting_data_end_date,
                freq='D',
                tz=pytz.UTC)

        reporting_period_daily_fixture = formatter.create_demand_fixture(
            reporting_period_daily_index, weather_source)
        reporting_period_fixture_success = True
        if len(reporting_period_daily_fixture) == 0:
            reporting_period_fixture_success = False

        if baseline_data_start_date == baseline_data_end_date:
            baseline_period_daily_index = pd.Series([])
        else:
            baseline_period_daily_index = pd.date_range(
                start=baseline_data_start_date,
                end=baseline_data_end_date,
                freq='D',
                tz=pytz.UTC)

        baseline_period_daily_fixture = formatter.create_demand_fixture(
            baseline_period_daily_index, weather_source)
        baseline_period_fixture_success = True
        if len(baseline_period_daily_fixture) == 0:
            baseline_period_fixture_success = False

        # Apply mask which indicates where data is missing (with daily
        # resolution)
        unmasked_reporting_period_daily_fixture = \
            reporting_period_daily_fixture.copy()
        if 'input_mask' in reporting_output.keys():
            reporting_mask = reporting_output['input_mask']
            for i, mask in reporting_mask.iteritems():
                if pd.isnull(mask):
                    reporting_period_daily_fixture[i] = np.nan
        else:
            reporting_mask = pd.Series([])

        unmasked_baseline_period_daily_fixture = \
            baseline_period_daily_fixture.copy()
        if 'input_mask' in baseline_output.keys():
            baseline_mask = baseline_output['input_mask']
            for i, mask in baseline_mask.iteritems():
                if pd.isnull(mask):
                    baseline_period_daily_fixture[i] = np.nan
        else:
            baseline_mask = pd.Series([])

    else:
        reporting_mask = pd.Series([])
        baseline_mask = pd.Series([])
        baseline_period_daily_fixture = None
        reporting_period_daily_fixture = None
        unmasked_baseline_period_daily_fixture = None
        unmasked_reporting_period_daily_fixture = None
        baseline_period_fixture_success = False
        reporting_period_fixture_success = False
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
            'annualized_daily_fixture': annualized_daily_fixture,
            'baseline_model': baseline_model,
            'reporting_model': reporting_model,
            'baseline_period_daily_fixture': baseline_period_daily_fixture,
            'baseline_period_fixture_success': baseline_period_fixture_success,
            'reporting_period_daily_fixture': reporting_period_daily_fixture,
            'reporting_period_fixture_success': reporting_period_fixture_success,
            'baseline_mask': baseline_mask,
            'reporting_mask': reporting_mask,
            'unmasked_baseline_period_daily_fixture': unmasked_baseline_period_daily_fixture,
            'unmasked_reporting_period_daily_fixture': unmasked_reporting_period_daily_fixture,
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
            deriv_input['baseline_model'].predict(deriv_input['annualized_daily_fixture'], summed=True),
            deriv_input['reporting_model'].predict(deriv_input['annualized_daily_fixture'], summed=True))
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
            deriv_input['baseline_model'].predict(deriv_input['annualized_daily_fixture'], summed=False),
            deriv_input['reporting_model'].predict(deriv_input['annualized_daily_fixture'], summed=False))
        return {
                'series': series,
                'description': description,
                'orderable': [i.isoformat() for i in deriv_input['annualized_daily_fixture'].index],
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
                          deriv_input['annualized_daily_fixture'], summed=True)
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
                    deriv_input['annualized_daily_fixture'], summed=False)
        return {
                'series': series,
                'description': description,
                'orderable': [i.isoformat() for i in deriv_input['annualized_daily_fixture'].index],
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
                          deriv_input['reporting_period_daily_fixture'], summed=True)
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
                          deriv_input['reporting_period_daily_fixture'], summed=False)
        return {
                'series': series,
                'description': description,
                'orderable': [i.isoformat() for i in deriv_input['reporting_period_daily_fixture'].index],
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
                          deriv_input['reporting_period_daily_fixture'], summed=False)
        return {
                'series': series,
                'description': description,
                'orderable': [i.isoformat() for i in deriv_input['reporting_period_daily_fixture'].index],
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
                    deriv_input['reporting_period_daily_fixture'], summed=True),
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
                    deriv_input['reporting_period_daily_fixture'], summed=False),
                (deriv_input['reporting_period_data'], 0)
            )

        return {
                'series': series,
                'description': description,
                'orderable': [i.isoformat() for i in deriv_input['reporting_period_daily_fixture'].index],
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
                    deriv_input['reporting_period_daily_fixture'], summed=False),
                (deriv_input['reporting_period_data'], 0)
            )
        return {
                'series': series,
                'description': description,
                'orderable': [i.isoformat() for i in deriv_input['reporting_period_daily_fixture'].index],
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
                          deriv_input['baseline_period_daily_fixture'], summed=False)
        return {
                'series': series,
                'description': description,
                'orderable': [i.isoformat() for i in deriv_input['baseline_period_daily_fixture'].index],
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
                    deriv_input['annualized_daily_fixture'], summed=True)
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
                    deriv_input['annualized_daily_fixture'], summed=False)
        return {
                'series': series,
                'description': description,
                'orderable': [i.isoformat() for i in deriv_input['annualized_daily_fixture'].index],
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
                deriv_input['reporting_period_daily_fixture'], summed=False)
        return {
                'series': series,
                'description': description,
                'orderable': [i.isoformat() for i in deriv_input['reporting_period_daily_fixture'].index],
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
                    i.isoformat() for i in deriv_input['unmasked_baseline_period_daily_fixture'].index],
                'value': deriv_input['unmasked_baseline_period_daily_fixture']['tempF'].values.tolist(),
                'variance': [0 for _ in range(deriv_input['unmasked_baseline_period_daily_fixture']['tempF'].shape[0])]
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
                    i.isoformat() for i in deriv_input['unmasked_reporting_period_daily_fixture'].index],
                'value': deriv_input['unmasked_reporting_period_daily_fixture']['tempF'].values.tolist(),
                'variance': [0 for _ in range(deriv_input['unmasked_reporting_period_daily_fixture']['tempF'].shape[0])]
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
                    i.isoformat() for i in deriv_input['unmasked_reporting_period_daily_fixture'].index],
                'value': [
                    v if not deriv_input['reporting_mask'].get(i, True) else None
                    for i, v in deriv_input['unmasked_reporting_period_daily_fixture']['tempF'].iteritems()
                ],
                'variance': [
                    0 if not deriv_input['reporting_mask'].get(i, True) else None
                    for i, v in deriv_input['unmasked_reporting_period_daily_fixture']['tempF'].iteritems()
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
                    i.isoformat() for i in deriv_input['annualized_daily_fixture'].index],
                'value': deriv_input['annualized_daily_fixture']['tempF'].values.tolist(),
                'variance': [0 for _ in range(deriv_input['annualized_daily_fixture']['tempF'].shape[0])]
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
