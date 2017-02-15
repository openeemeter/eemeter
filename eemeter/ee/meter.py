import logging
from collections import OrderedDict, namedtuple

import numpy as np
import pandas as pd
import pytz

from eemeter import get_version
from eemeter.modeling.formatters import (
    ModelDataFormatter,
    ModelDataBillingFormatter,
)
from eemeter.modeling.models import CaltrackMonthlyModel
from eemeter.modeling.split import SplitModeledEnergyTrace
from eemeter.io.serializers import (
    deserialize_meter_input,
    serialize_derivatives,
    serialize_split_modeled_energy_trace,
)
from eemeter.processors.dispatchers import (
    get_approximate_frequency,
)
from eemeter.processors.location import (
    get_weather_normal_source,
    get_weather_source,
)
from eemeter.structures import ZIPCodeSite

logger = logging.getLogger(__name__)


Derivative = namedtuple('Derivative', [
    'modeling_period_group',
    'source',
    'series',
    'orderable',
    'value',
    'variance',
    'unit',
    'serialized_demand_fixture'
])


class EnergyEfficiencyMeter(object):
    ''' Meter for determining energy efficiency derivatives for a single
    traces.

    Parameters
    ----------
    default_model_mapping : dict
        mapping between (interpretation, frequency) tuples used to select
        the default model (if none is explicitly provided in `.evaluate()`).

    '''

    def __init__(self, default_model_mapping=None,
                 default_formatter_mapping=None):

        if default_formatter_mapping is None:
            daily_formatter = (ModelDataFormatter, {'freq_str': 'D'})
            billing_formatter = (ModelDataBillingFormatter, {})
            default_formatter_mapping = {
                ('NATURAL_GAS_CONSUMPTION_SUPPLIED', '15T'):
                    daily_formatter,
                ('ELECTRICITY_CONSUMPTION_SUPPLIED', '15T'):
                    daily_formatter,
                ('ELECTRICITY_ON_SITE_GENERATION_UNCONSUMED', '15T'):
                    daily_formatter,

                ('NATURAL_GAS_CONSUMPTION_SUPPLIED', '30T'):
                    daily_formatter,
                ('ELECTRICITY_CONSUMPTION_SUPPLIED', '30T'):
                    daily_formatter,
                ('ELECTRICITY_ON_SITE_GENERATION_UNCONSUMED', '30T'):
                    daily_formatter,

                ('NATURAL_GAS_CONSUMPTION_SUPPLIED', 'H'):
                    daily_formatter,
                ('ELECTRICITY_CONSUMPTION_SUPPLIED', 'H'):
                    daily_formatter,
                ('ELECTRICITY_ON_SITE_GENERATION_UNCONSUMED', 'H'):
                    daily_formatter,

                ('NATURAL_GAS_CONSUMPTION_SUPPLIED', 'D'):
                    daily_formatter,
                ('ELECTRICITY_CONSUMPTION_SUPPLIED', 'D'):
                    daily_formatter,
                ('ELECTRICITY_ON_SITE_GENERATION_UNCONSUMED', 'D'):
                    daily_formatter,

                ('NATURAL_GAS_CONSUMPTION_SUPPLIED', None):
                    billing_formatter,
                ('ELECTRICITY_CONSUMPTION_SUPPLIED', None):
                    billing_formatter,
            }

        if default_model_mapping is None:
            caltrack_gas_model = (CaltrackMonthlyModel, {
                'fit_cdd': False,
                'grid_search': True,
            })
            caltrack_elec_model = (CaltrackMonthlyModel, {
                'fit_cdd': True,
                'grid_search': True,
            })
            default_model_mapping = {
                ('NATURAL_GAS_CONSUMPTION_SUPPLIED', '15T'):
                    caltrack_gas_model,
                ('ELECTRICITY_CONSUMPTION_SUPPLIED', '15T'):
                    caltrack_elec_model,
                ('ELECTRICITY_ON_SITE_GENERATION_UNCONSUMED', '15T'):
                    caltrack_elec_model,

                ('NATURAL_GAS_CONSUMPTION_SUPPLIED', '30T'):
                    caltrack_gas_model,
                ('ELECTRICITY_CONSUMPTION_SUPPLIED', '30T'):
                    caltrack_elec_model,
                ('ELECTRICITY_ON_SITE_GENERATION_UNCONSUMED', '30T'):
                    caltrack_elec_model,

                ('NATURAL_GAS_CONSUMPTION_SUPPLIED', 'H'):
                    caltrack_gas_model,
                ('ELECTRICITY_CONSUMPTION_SUPPLIED', 'H'):
                    caltrack_elec_model,
                ('ELECTRICITY_ON_SITE_GENERATION_UNCONSUMED', 'H'):
                    caltrack_elec_model,

                ('NATURAL_GAS_CONSUMPTION_SUPPLIED', 'D'):
                    caltrack_gas_model,
                ('ELECTRICITY_CONSUMPTION_SUPPLIED', 'D'):
                    caltrack_elec_model,
                ('ELECTRICITY_ON_SITE_GENERATION_UNCONSUMED', 'D'):
                    caltrack_elec_model,

                ('NATURAL_GAS_CONSUMPTION_SUPPLIED', None):
                    caltrack_gas_model,
                ('ELECTRICITY_CONSUMPTION_SUPPLIED', None):
                    caltrack_elec_model,
            }

        self.default_formatter_mapping = default_formatter_mapping
        self.default_model_mapping = default_model_mapping

    def evaluate(self, meter_input, formatter=None,
                 model=None, weather_source=None, weather_normal_source=None):
        ''' Main entry point to the meter, which models traces and calculates
        derivatives.

        Parameters
        ----------
        trace : eemeter.structures.EnergyTrace
            Trace for which to evaluate meter
        site : eemeter.structures.ZIPCodeSite
            Contains ZIP code to match to weather stations from which to pull
            weather data.
        modeling_period_set : eemeter.structures.ModelPeriodSet
            Modeling periods to use in evaluation.
        formatter : tuple of (class, dict), default None
            Formatter for trace and weather data. Used to create input
            for model. If None is provided, will be auto-matched to appropriate
            default formatter.
        model : tuple of (class, dict), default None
            Model to use in modeling. If None is provided,
            will be auto-matched to appropriate default model.
        weather_source : eemeter.weather.WeatherSource
            Weather source to be used for this meter. Overrides weather source
            found using :code:`project.site`. Useful for test mocking.
        weather_normal_source : eemeter.weather.WeatherSource
            Weather normal source to be used for this meter. Overrides weather
            source found using :code:`project.site`. Useful for test mocking.

        Returns
        -------
        results : dict
            Dictionary of results with the following keys:

            - :code:`"status"`: SUCCESS/FAILURE
            - :code:`"failure_message"`: if FAILURE, message indicates reason
              for failure, may include traceback
            - :code:`"logs"`: list of collected log messages
            - :code:`"model_class"`: Name of model class
            - :code:`"model_kwargs"`: dict of model keyword arguments
              (settings)
            - :code:`"formatter_class"`: Name of formatter class
            - :code:`"formatter_kwargs"`: dict of formatter keyword arguments
              (settings)
            - :code:`"eemeter_version"`: version of the eemeter package
            - :code:`"modeled_energy_trace"`: modeled energy trace
            - :code:`"derivatives"`: derivatives for each interpretation
            - :code:`"weather_source_station"`: Matched weather source station.
            - :code:`"weather_normal_source_station"`: Matched weather normal
              source station.
        '''

        SUCCESS = "SUCCESS"
        FAILURE = "FAILURE"

        output = OrderedDict([
            ("status", None),
            ("failure_message", None),
            ("logs", []),

            ("eemeter_version", get_version()),

            ("model_class", None),
            ("model_kwargs", None),
            ("formatter_class", None),
            ("formatter_kwargs", None),

            ("weather_source_station", None),
            ("weather_normal_source_station", None),
            ("derivatives", None),
            ("modeled_energy_trace", None),
        ])

        # Step 1: Deserialize input and validate
        deserialized_input = deserialize_meter_input(meter_input)
        if "error" in deserialized_input:
            message = (
                "Meter input could not be deserialized:\n{}"
                .format(deserialized_input)
            )
            output['status'] = FAILURE
            output['failure_message'] = message
            return output

        # Assume that deserialized input fails without these keys, so don't
        # bother error checking
        trace = deserialized_input["trace"]
        project = deserialized_input["project"]
        zipcode = project["zipcode"]
        site = ZIPCodeSite(zipcode)

        # Can be blank for models capable of structural change analysis, so
        # provide default
        modeling_period_set = project.get("modeling_period_set", None)

        # Step 2: Match weather
        if weather_source is None:
            weather_source = get_weather_source(site)
            message = "Using weather_source {}".format(weather_source)
            output['weather_source_station'] = None
        else:
            message = "Using supplied weather_source"
            logger.info(message)
            output['weather_source_station'] = weather_source.station
        output['logs'].append(message)

        if weather_normal_source is None:
            weather_normal_source = get_weather_normal_source(site)
            message = (
                "Using weather_normal_source {}"
                .format(weather_normal_source)
            )
        else:
            message = "Using supplied weather_normal_source"
            logger.info(message)
        output['weather_normal_source_station'] = weather_normal_source.station
        output['logs'].append(message)

        # Step 3: Check to see if trace is placeholder. If so,
        # return with SUCCESS, empty derivatives.
        if trace.placeholder:
            message = (
                'Skipping modeling for placeholder trace {}'
                .format(trace)
            )
            logger.info(message)
            output['logs'].append(message)
            output['status'] = SUCCESS
            output['derivatives'] = []
            return output

        # Step 4: Determine trace interpretation and frequency
        if model is None or formatter is None:
            trace_interpretation = trace.interpretation
            trace_frequency = get_approximate_frequency(trace)

            if trace_frequency not in ['H', 'D', '15T', '30T']:
                trace_frequency = None

            selector = (trace_interpretation, trace_frequency)

        # Step 5: create formatter instance
        if formatter is None:
            FormatterClass, formatter_kwargs = self.default_formatter_mapping \
                .get(selector, (None, None))
            if FormatterClass is None:
                message = (
                    "Default formatter mapping did not find a match for"
                    " the selector {}".format(selector)
                )
                output['status'] = FAILURE
                output['failure_message'] = message
                return output
        else:
            FormatterClass, formatter_kwargs = formatter
        formatter_instance = FormatterClass(**formatter_kwargs)
        output["formatter_class"] = FormatterClass.__name__
        output["formatter_kwargs"] = formatter_kwargs

        # Step 6: create model instance
        if model is None:
            ModelClass, model_kwargs = self.default_model_mapping.get(
                selector, (None, None))
            if ModelClass is None:
                message = (
                    "Default model mapping did not find a match for the"
                    " selector {}".format(selector)
                )
                output['status'] == FAILURE
                output['failure_message'] = message
                return output
        else:
            ModelClass, model_kwargs = model
        output["model_class"] = ModelClass.__name__
        output["model_kwargs"] = model_kwargs

        # Step 7: validate modeling period set. Always fails for now, since
        # no models are yet fully structural change analysis aware
        if modeling_period_set is None:
            message = (
                "Model is not structural-change capable, so `modeling_period`"
                " argument must be supplied."
            )
            output['status'] == FAILURE
            output['failure_message'] = message
            return output

        # Step 8: create split modeled energy trace
        model_mapping = {
            modeling_period_label: ModelClass(\
                modeling_period_interpretation=modeling_period_label, \
                **model_kwargs)
            for modeling_period_label, _ in
            modeling_period_set.iter_modeling_periods()
        }

        modeled_trace = SplitModeledEnergyTrace(
            trace, formatter_instance, model_mapping, modeling_period_set)

        modeled_trace.fit(weather_source)
        output["modeled_energy_trace"] = \
            serialize_split_modeled_energy_trace(modeled_trace)

        # Step 9: for each modeling period group, create derivatives
        derivatives = []
        for ((baseline_label, reporting_label),
             (baseline_period, reporting_period)) in \
                modeling_period_set.iter_modeling_period_groups():

            baseline_output = modeled_trace.fit_outputs[baseline_label]
            reporting_output = modeled_trace.fit_outputs[reporting_label]

            baseline_model_success = (baseline_output["status"] == "SUCCESS")
            reporting_model_success = (reporting_output["status"] == "SUCCESS")

            formatter = modeled_trace.formatter
            unit = modeled_trace.trace.unit
            trace = modeled_trace.trace

            # annualized fixture
            normal_index = pd.date_range(
                '2015-01-01', freq='D', periods=365, tz=pytz.UTC)
            annualized_daily_fixture = formatter.create_demand_fixture(
                normal_index, weather_normal_source)

            # default project dates
            baseline_end_date = baseline_period.end_date
            reporting_start_date = reporting_period.start_date

            # Note: observed data uses project dates, not data dates
            # convert trace data to daily
            daily_trace_data = formatter.daily_trace_data(trace)
            if daily_trace_data.empty:
                continue

            baseline_period_data = daily_trace_data[:baseline_end_date].copy()
            project_period_data = \
                daily_trace_data[baseline_end_date:reporting_start_date].copy()
            reporting_period_data = \
                daily_trace_data[reporting_start_date:].copy()

            # get monthly versions of the daily data
            def by_month(series):
                if series.empty:
                    return []
                return [
                    ("{}-{:02d}".format(year, month), group)
                    for (year, month), group in pd.groupby(
                        series, by=[series.index.year, series.index.month])
                ]

            annualized_daily_fixture_monthly = \
                by_month(annualized_daily_fixture)
            observed_baseline_period_monthly = by_month(baseline_period_data)
            observed_project_period_monthly = by_month(project_period_data)
            observed_reporting_period_monthly = by_month(reporting_period_data)

            # find start and end dates of reporting data
            if not reporting_period_data.empty:
                reporting_data_start_date = reporting_period_data.index[0]
                reporting_data_end_date = reporting_period_data.index[-1]
            else:
                reporting_data_start_date = reporting_start_date
                reporting_data_end_date = reporting_start_date

            baseline_model = modeled_trace.model_mapping[baseline_label]
            reporting_model = modeled_trace.model_mapping[reporting_label]

            # reporting period fixture
            if None not in (
                    reporting_data_start_date, reporting_data_end_date):

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

                # Apply mask which indicates where data is missing (with daily
                # resolution)
                if 'input_mask' in reporting_output.keys():
                    reporting_mask = reporting_output['input_mask']
                    for i, mask in reporting_mask.iteritems():
                        if pd.isnull(mask):
                            reporting_period_daily_fixture[i] = np.nan

                reporting_period_daily_fixture_monthly = \
                    by_month(reporting_period_daily_fixture)

                # do some month realignment to make sure they match
                fixture_months = set([
                    m for (m, _) in reporting_period_daily_fixture_monthly
                ])
                observed_months = set([
                    m for (m, _) in observed_reporting_period_monthly
                ])

                all_months = fixture_months | observed_months
                common_months = fixture_months & observed_months
                misaligned_months = all_months - common_months
                if len(misaligned_months) > 0:
                    message = (
                        'Derivative orderables {} missing from one of'
                        'daily_fixture or observed series'
                        .format(misaligned_months)
                    )
                    logger.warning(message)

                reporting_period_daily_fixture_monthly = [
                    (m, d)
                    for (m, d) in reporting_period_daily_fixture_monthly
                    if m in common_months
                ]

                observed_reporting_period_monthly = [
                    (m, d)
                    for (m, d) in observed_reporting_period_monthly
                    if m in common_months
                ]

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

            raw_derivatives = []

            serialize_demand_fixture = formatter.serialize_demand_fixture

            def serialize_observed(series):
                return OrderedDict([
                    (start.isoformat(), value)
                    for start, value in series.iteritems()
                ])

            def _report_failed_derivative(source, series, orderable):
                logger.warning(
                    'Failed computing derivative (source={}, series={}, orderable={})'
                    .format(source, series, orderable)
                )

            if baseline_model_success and reporting_model_success:

                source = 'baseline_model_minus_reporting_model'
                series = 'annualized_weather_normal'
                orderable = None
                try:
                    raw_derivatives.append({
                        'source': source,
                        'series': series,
                        'orderable': orderable,
                        'serialized_demand_fixture': serialize_demand_fixture(
                            annualized_daily_fixture),
                        'value_variance': subtract_value_variance_tuple(
                            baseline_model.predict(
                                annualized_daily_fixture, summed=True),
                            reporting_model.predict(
                                annualized_daily_fixture, summed=True),
                        ),
                    })
                except:
                    _report_failed_derivative(source, series, orderable)

                for (month, values) in annualized_daily_fixture_monthly:

                    source = 'baseline_model_minus_reporting_model'
                    series = 'annualized_weather_normal_monthly'
                    orderable = month
                    try:
                        raw_derivatives.append({
                            'source': source,
                            'series': series,
                            'orderable': orderable,
                            'serialized_demand_fixture':
                                serialize_demand_fixture(values),
                            'value_variance': subtract_value_variance_tuple(
                                baseline_model.predict(values, summed=True),
                                reporting_model.predict(values, summed=True)),
                        })
                    except:
                        _report_failed_derivative(source, series, orderable)

            if baseline_model_success:

                source = 'baseline_model'
                series = 'annualized_weather_normal'
                orderable =  None
                try:
                    raw_derivatives.append({
                        'source': source,
                        'series': series,
                        'orderable': orderable,
                        'serialized_demand_fixture':
                            serialize_demand_fixture(annualized_daily_fixture),
                        'value_variance':
                            baseline_model.predict(
                                annualized_daily_fixture, summed=True),
                    })
                except:
                    _report_failed_derivative(source, series, orderable)

                source = 'baseline_model'
                series = 'reporting_cumulative'
                orderable = None
                try:
                    raw_derivatives.append({
                        'source': source,
                        'series': series,
                        'orderable': orderable,
                        'serialized_demand_fixture': serialize_demand_fixture(
                            reporting_period_daily_fixture),
                        'value_variance': baseline_model.predict(
                            reporting_period_daily_fixture, summed=True),
                    })
                except:
                    _report_failed_derivative(source, series, orderable)

                source = 'baseline_model_minus_observed'
                series = 'reporting_cumulative'
                orderable = None
                try:
                    raw_derivatives.append({
                        'source': source,
                        'series': series,
                        'orderable': orderable,
                        'serialized_demand_fixture': (
                            serialize_demand_fixture(
                                reporting_period_daily_fixture),
                            serialize_observed(reporting_period_data)
                        ),
                        'value_variance': subtract_value_variance_tuple(
                            baseline_model.predict(
                                reporting_period_daily_fixture, summed=True),
                            (reporting_period_data.sum(), 0),
                        ),
                    })
                except:
                    _report_failed_derivative(source, series, orderable)

                for (month, values) in annualized_daily_fixture_monthly:
                    source = 'baseline_model'
                    series = 'annualized_weather_normal_monthly'
                    orderable = month
                    try:
                        raw_derivatives.append({
                            'source': source,
                            'series': series,
                            'orderable': orderable,
                            'serialized_demand_fixture':
                                serialize_demand_fixture(values),
                            'value_variance':
                                baseline_model.predict(values, summed=True),
                        })
                    except:
                        _report_failed_derivative(source, series, orderable)

                for (month, values) in reporting_period_daily_fixture_monthly:
                    source = 'baseline_model'
                    series = 'reporting_monthly'
                    orderable = month
                    try:
                        raw_derivatives.append({
                            'source': source,
                            'series': series,
                            'orderable': orderable,
                            'serialized_demand_fixture':
                                serialize_demand_fixture(values),
                            'value_variance':
                                baseline_model.predict(values, summed=True),
                        })
                    except:
                        _report_failed_derivative(source, series, orderable)

                for (month1, values1), (month2, values2) in zip(
                        reporting_period_daily_fixture_monthly,
                        observed_reporting_period_monthly):
                    source = 'baseline_model_minus_observed'
                    series = 'reporting_monthly'
                    orderable =  month1  # == month2; we checked above
                    try:
                        raw_derivatives.append({
                            'source': source,
                            'series': series,
                            'orderable': orderable,
                            'serialized_demand_fixture': (
                                serialize_demand_fixture(values1),
                                serialize_observed(values2)
                            ),
                            'value_variance': subtract_value_variance_tuple(
                                baseline_model.predict(values1, summed=True),
                                (values2.sum(), 0),
                            ),
                        })
                    except:
                        _report_failed_derivative(source, series, orderable)

            if reporting_model_success:

                source = 'reporting_model'
                series = 'annualized_weather_normal'
                orderable = None
                try:
                    raw_derivatives.append({
                        'source': source,
                        'series': series,
                        'orderable': orderable,
                        'serialized_demand_fixture':
                            serialize_demand_fixture(annualized_daily_fixture),
                        'value_variance':
                            reporting_model.predict(
                                annualized_daily_fixture, summed=True),
                    })
                except:
                    _report_failed_derivative(source, series, orderable)

                for (month, values) in annualized_daily_fixture_monthly:
                    source = 'reporting_model'
                    series = 'annualized_weather_normal_monthly'
                    orderable = month
                    try:
                        raw_derivatives.append({
                            'source': source,
                            'series': series,
                            'orderable': orderable,
                            'serialized_demand_fixture':
                                serialize_demand_fixture(values),
                            'value_variance':
                                reporting_model.predict(values, summed=True),
                        })
                    except:
                        _report_failed_derivative(source, series, orderable)

            source = 'observed'
            series = 'reporting_cumulative'
            orderable = None
            try:
                raw_derivatives.append({
                    'source': source,
                    'series': series,
                    'orderable': orderable,
                    'serialized_demand_fixture':
                        serialize_observed(reporting_period_data),
                    'value_variance': (reporting_period_data.sum(), 0),
                })
            except:
                _report_failed_derivative(source, series, orderable)

            for (month, values) in observed_reporting_period_monthly:
                source = 'observed'
                series = 'reporting_monthly'
                orderable = month
                try:
                    raw_derivatives.append({
                        'source': source,
                        'series': series,
                        'orderable': orderable,
                        'serialized_demand_fixture': serialize_observed(values),
                        'value_variance': (values.sum(), 0),
                    })
                except:
                    _report_failed_derivative(source, series, orderable)

            for (month, values) in observed_baseline_period_monthly:
                source = 'observed'
                series = 'baseline_monthly'
                orderable = month
                try:
                    raw_derivatives.append({
                        'source': source,
                        'series': series,
                        'orderable': orderable,
                        'serialized_demand_fixture': serialize_observed(values),
                        'value_variance': (values.sum(), 0),
                    })
                except:
                    _report_failed_derivative(source, series, orderable)

            for (month, values) in observed_project_period_monthly:
                source = 'observed'
                series = 'project_monthly'
                orderable = month
                try:
                    raw_derivatives.append({
                        'source': source,
                        'series': series,
                        'orderable': orderable,
                        'serialized_demand_fixture': serialize_observed(values),
                        'value_variance': (values.sum(), 0),
                    })
                except:
                    _report_failed_derivative(source, series, orderable)

            derivatives += [
                Derivative(
                    (baseline_label, reporting_label),
                    d['source'],
                    d['series'],
                    d['orderable'],
                    d['value_variance'][0],
                    d['value_variance'][1],
                    unit,
                    d['serialized_demand_fixture']
                )
                for d in raw_derivatives
            ]

        output["derivatives"] = serialize_derivatives(derivatives)
        output["status"] = SUCCESS
        return output
