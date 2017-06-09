import logging
from collections import OrderedDict, namedtuple

from six import string_types
import numpy as np
import pandas as pd
import pytz
from functools import reduce

from eemeter import get_version
from eemeter.modeling.formatters import (
    ModelDataFormatter,
    ModelDataBillingFormatter,
)
from eemeter.modeling.models import CaltrackMonthlyModel, CaltrackDailyModel
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
    'series',
    'description',
    'orderable',
    'value',
    'variance'
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
            caltrack_gas_model_daily = (CaltrackDailyModel, {
                'fit_cdd': False,
                'grid_search': True,
            })
            caltrack_elec_model_daily = (CaltrackDailyModel, {
                'fit_cdd': True,
                'grid_search': True,
            })
            default_model_mapping = {
                ('NATURAL_GAS_CONSUMPTION_SUPPLIED', '15T'):
                    caltrack_gas_model_daily,
                ('ELECTRICITY_CONSUMPTION_SUPPLIED', '15T'):
                    caltrack_elec_model_daily,
                ('ELECTRICITY_ON_SITE_GENERATION_UNCONSUMED', '15T'):
                    caltrack_elec_model_daily,

                ('NATURAL_GAS_CONSUMPTION_SUPPLIED', '30T'):
                    caltrack_gas_model_daily,
                ('ELECTRICITY_CONSUMPTION_SUPPLIED', '30T'):
                    caltrack_elec_model_daily,
                ('ELECTRICITY_ON_SITE_GENERATION_UNCONSUMED', '30T'):
                    caltrack_elec_model_daily,

                ('NATURAL_GAS_CONSUMPTION_SUPPLIED', 'H'):
                    caltrack_gas_model_daily,
                ('ELECTRICITY_CONSUMPTION_SUPPLIED', 'H'):
                    caltrack_elec_model_daily,
                ('ELECTRICITY_ON_SITE_GENERATION_UNCONSUMED', 'H'):
                    caltrack_elec_model_daily,

                ('NATURAL_GAS_CONSUMPTION_SUPPLIED', 'D'):
                    caltrack_gas_model_daily,
                ('ELECTRICITY_CONSUMPTION_SUPPLIED', 'D'):
                    caltrack_elec_model_daily,
                ('ELECTRICITY_ON_SITE_GENERATION_UNCONSUMED', 'D'):
                    caltrack_elec_model_daily,

                ('NATURAL_GAS_CONSUMPTION_SUPPLIED', None):
                    caltrack_gas_model,
                ('ELECTRICITY_CONSUMPTION_SUPPLIED', None):
                    caltrack_elec_model,
            }

        self.default_formatter_mapping = default_formatter_mapping
        self.default_model_mapping = default_model_mapping

    def _get_formatter(self, formatter, selector):
        # get the default mappings
        default_formatter_class, default_formatter_kwargs = \
            self.default_formatter_mapping.get(selector, (None, None))

        if formatter is None:
            # use defaults
            FormatterClass = default_formatter_class
            formatter_kwargs = default_formatter_kwargs

        # Use any info provided
        else:
            custom_formatter_class, custom_formatter_kwargs = formatter

            if custom_formatter_class is None:
                # use default formatter
                FormatterClass = default_formatter_class

                if custom_formatter_kwargs is None:
                    formatter_kwargs = default_formatter_kwargs
                else:
                    formatter_kwargs = default_formatter_kwargs
                    formatter_kwargs.update(custom_formatter_kwargs)
            else:
                # use custom formatter, which may be a string.
                if isinstance(custom_formatter_class, string_types):
                    FormatterClass = {
                        f.__name__: f
                        for f in [ModelDataFormatter, ModelDataBillingFormatter]
                    }[custom_formatter_class]
                else:
                    FormatterClass = custom_formatter_class

                if custom_formatter_kwargs is None:
                    # assume default args don't apply since using custom meter class
                    formatter_kwargs = {}
                else:
                    formatter_kwargs = custom_formatter_kwargs

        return FormatterClass, formatter_kwargs

    def _get_model(self, model, selector):
        # get the default mappings
        default_model_class, default_model_kwargs = \
            self.default_model_mapping.get(selector, (None, None))

        if model is None:
            # use defaults
            ModelClass = default_model_class
            model_kwargs = default_model_kwargs
        # Use any info provided
        else:
            custom_model_class, custom_model_kwargs = model

            if custom_model_class is None:
                # use default model
                ModelClass = default_model_class

                if custom_model_kwargs is None:
                    model_kwargs = default_model_kwargs
                else:
                    model_kwargs = default_model_kwargs
                    model_kwargs.update(custom_model_kwargs)
            else:
                # use custom model, which may be a string.
                if isinstance(custom_model_class, string_types):
                    ModelClass = {
                        f.__name__: f
                        for f in [CaltrackMonthlyModel]
                    }[custom_model_class]
                else:
                    ModelClass = custom_model_class

                if custom_model_kwargs is None:
                    # assume default args don't apply since using custom meter class
                    model_kwargs = {}
                else:
                    model_kwargs = custom_model_kwargs

        return ModelClass, model_kwargs

    def evaluate(self, meter_input, formatter=None,
                 model=None, weather_source=None, weather_normal_source=None):
        ''' Main entry point to the meter, which models traces and calculates
        derivatives.

        Parameters
        ----------
        meter_input : dict
            Serialized input containing trace and project data.
        formatter : tuple of (class, dict), default None
            Formatter for trace and weather data. Used to create input
            for model. If None is provided, will be auto-matched to appropriate
            default formatter. Class name can be provided as a string
            (class.__name__) or object.
        model : tuple of (class, dict), default None
            Model to use in modeling. If None is provided,
            will be auto-matched to appropriate default model.
            Class can be provided as a string (class.__name__) or class object.
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
            ("trace_id", None),
            ("project_id", None),
            ("interval", None),

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

        project_id = project["project_id"]
        trace_id = trace.trace_id
        interval = trace.interval

        output['project_id'] = project_id
        output['trace_id'] = trace_id
        output['interval'] = interval

        logger.debug(
            'Running meter for for trace {} and project {}'
            .format(project_id, trace_id)
        )

        # Step 2: Match weather
        if weather_source is None:
            weather_source = get_weather_source(site)
            if weather_source is None:
                message = (
                    "Could not find weather normal source matching site {}"
                    .format(site)
                )
                weather_source_station = None
            else:
                message = "Using weather_source {}".format(weather_source)
                weather_source_station = weather_source.station
        else:
            message = "Using supplied weather_source"
            weather_source_station = weather_source.station
        output['weather_source_station'] = weather_source_station
        output['logs'].append(message)
        logger.debug(message)

        if weather_normal_source is None:
            weather_normal_source = get_weather_normal_source(site)
            if weather_normal_source is None:
                message = (
                    "Could not find weather normal source matching site {}"
                    .format(site)
                )
                weather_normal_source_station = None
            else:
                message = (
                    "Using weather_normal_source {}"
                    .format(weather_normal_source)
                )
                weather_normal_source_station = weather_normal_source.station
        else:
            message = "Using supplied weather_normal_source"
            weather_normal_source_station = weather_normal_source.station
        output['weather_normal_source_station'] = weather_normal_source_station
        output['logs'].append(message)
        logger.debug(message)

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
        # TODO use trace interval here. And enforce upstream that interval use
        # pandas interval strings?
        trace_frequency = get_approximate_frequency(trace)

        if trace_frequency not in ['H', 'D', '15T', '30T']:
            trace_frequency = None

        selector = (trace.interpretation, trace_frequency)

        # Step 5: create formatter instance
        FormatterClass, formatter_kwargs = self._get_formatter(formatter, selector)
        if FormatterClass is None:
            message = (
                "Default formatter mapping did not find a match for the"
                " selector {}".format(selector)
            )
            output['status'] = FAILURE
            output['failure_message'] = message
            return output
        output["formatter_class"] = FormatterClass.__name__
        output["formatter_kwargs"] = formatter_kwargs
        formatter_instance = FormatterClass(**formatter_kwargs)

        # Step 6: create model instance
        ModelClass, model_kwargs = self._get_model(model, selector)
        if ModelClass is None:
            message = (
                "Default model mapping did not find a match for the"
                " selector {}".format(selector)
            )
            output['status'] = FAILURE
            output['failure_message'] = message
            return output
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
            modeling_period_label: ModelClass(
                modeling_period_interpretation=modeling_period_label,
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

            weather_source_success = (weather_source is not None)
            weather_normal_source_success = (weather_normal_source is not None)

            # annualized fixture
            if weather_normal_source_success:
                normal_index = pd.date_range(
                    '2015-01-01', freq='D', periods=365, tz=pytz.UTC)
                annualized_daily_fixture = formatter.create_demand_fixture(
                    normal_index, weather_normal_source)

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

            def _report_failed_derivative(series):
                logger.warning(
                    'Failed computing derivative (series={})'
                    .format(series)
                )

            if baseline_model_success:

                if 'model_fit' in baseline_output.keys() and \
                   'model_params' in baseline_output['model_fit'] and \
                   'hdd_bp' in baseline_output['model_fit']['model_params']:

                    series = 'Heating degree day balance point, baseline period'
                    description = '''Best-fit heating degree day balance point,
                                     if any, for baseline model'''
                    value = baseline_output['model_fit']['model_params']['hdd_bp']

                    raw_derivatives.append({
                            'series': series,
                            'description': description,
                            'orderable': [None,],
                            'value': [value,],
                            'variance': [None,]
                    })

                    if 'coefficients' in baseline_output['model_fit']['model_params'] and \
                       'HDD_' + str(value) in baseline_output['model_fit']['model_params']['coefficients']:

                        series = 'Best-fit heating coefficient, baseline period'
                        description = '''Best-fit heating coefficient,
                                         if any, for baseline model'''
                        value = baseline_output['model_fit']['model_params']['coefficients']\
                                               ['HDD_' + str(value)]

                        raw_derivatives.append({
                                'series': series,
                                'description': description,
                                'orderable': [None,],
                                'value': [value,],
                                'variance': [None,]
                        })

                if 'model_fit' in baseline_output and \
                   'model_params' in baseline_output['model_fit'] and \
                   'cdd_bp' in baseline_output['model_fit']['model_params']:
                    series = 'Cooling degree day balance point, baseline period'
                    description = '''Best-fit cooling degree day balance point,
                                     if any, for baseline model'''
                    value = baseline_output['model_fit']['model_params']['cdd_bp']

                    raw_derivatives.append({
                            'series': series,
                            'description': description,
                            'orderable': [None,],
                            'value': [value,],
                            'variance': [None,]
                    })

                    if 'coefficients' in baseline_output['model_fit']['model_params'] and \
                       'CDD_' + str(value) in baseline_output['model_fit']['model_params']['coefficients']:
                        series = 'Best-fit cooling coefficient, baseline period'
                        description = '''Best-fit cooling coefficient,
                                         if any, for baseline model'''
                        value = baseline_output['model_fit']['model_params']['coefficients']\
                                               ['CDD_' + str(value)]

                        raw_derivatives.append({
                                'series': series,
                                'description': description,
                                'orderable': [None,],
                                'value': [value,],
                                'variance': [None,]
                        })

                if 'model_fit' in baseline_output and \
                   'model_params' in baseline_output['model_fit'] and \
                   'coefficients' in baseline_output['model_fit']['model_params'] and \
                   'Intercept' in baseline_output['model_fit']['model_params']['coefficients']:
                    series = 'Best-fit intercept, baseline period'
                    description = '''Best-fit intercept, if any, for baseline model'''
                    value = baseline_output['model_fit']['model_params']['coefficients']['Intercept']

                    raw_derivatives.append({
                            'series': series,
                            'description': description,
                            'orderable': [None,],
                            'value': [value,],
                            'variance': [None,]
                    })

            if reporting_model_success:

                if 'model_fit' in reporting_output.keys() and \
                   'model_params' in reporting_output['model_fit'] and \
                   'hdd_bp' in reporting_output['model_fit']['model_params']:

                    series = 'Heating degree day balance point, reporting period'
                    description = '''Best-fit heating degree day balance point,
                                     if any, for reporting model'''
                    value = reporting_output['model_fit']['model_params']['hdd_bp']

                    raw_derivatives.append({
                            'series': series,
                            'description': description,
                            'orderable': [None,],
                            'value': [value,],
                            'variance': [None,]
                    })

                    if 'coefficients' in reporting_output['model_fit']['model_params'] and \
                       'HDD_' + str(value) in reporting_output['model_fit']['model_params']['coefficients']:

                        series = 'Best-fit heating coefficient, reporting period'
                        description = '''Best-fit heating coefficient,
                                         if any, for reporting model'''
                        value = reporting_output['model_fit']['model_params']['coefficients']\
                                               ['HDD_' + str(value)]
                        raw_derivatives.append({
                                'series': series,
                                'description': description,
                                'orderable': [None,],
                                'value': [value,],
                                'variance': [None,]
                        })

                if 'model_fit' in reporting_output and \
                   'model_params' in reporting_output['model_fit'] and \
                   'cdd_bp' in reporting_output['model_fit']['model_params']:

                    series = 'Cooling degree day balance point, reporting period'
                    description = '''Best-fit cooling degree day balance point,
                                     if any, for reporting model'''
                    value = reporting_output['model_fit']['model_params']['cdd_bp']

                    raw_derivatives.append({
                            'series': series,
                            'description': description,
                            'orderable': [None,],
                            'value': [value,],
                            'variance': [None,]
                    })

                    if 'coefficients' in reporting_output['model_fit']['model_params'] and \
                       'CDD_' + str(value) in reporting_output['model_fit']['model_params']['coefficients']:
                        series = 'Best-fit cooling coefficient, reporting period'
                        description = '''Best-fit cooling coefficient,
                                         if any, for reporting model'''
                        value = reporting_output['model_fit']['model_params']['coefficients']\
                                               ['CDD_' + str(value)]

                        raw_derivatives.append({
                                'series': series,
                                'description': description,
                                'orderable': [None,],
                                'value': [value,],
                                'variance': [None,]
                        })

                if 'model_fit' in reporting_output and \
                   'model_params' in reporting_output['model_fit'] and \
                   'coefficients' in reporting_output['model_fit']['model_params'] and \
                   'Intercept' in reporting_output['model_fit']['model_params']['coefficients']:
                    series = 'Best-fit intercept, reporting period'
                    description = '''Best-fit intercept, if any, for reporting model'''
                    value = reporting_output['model_fit']['model_params']['coefficients']['Intercept']

                    raw_derivatives.append({
                            'series': series,
                            'description': description,
                            'orderable': [None,],
                            'value': [value,],
                            'variance': [None,]
                    })

            if baseline_model_success and reporting_model_success \
                    and weather_normal_source_success:
                series = 'Cumulative baseline model minus reporting model, normal year'
                description = '''Total predicted usage according to the baseline model
                                 over the normal weather year, minus the total predicted
                                 usage according to the reporting model over the normal
                                 weather year. Days for which normal year weather data
                                 does not exist are removed.'''
                try:
                    value, variance = subtract_value_variance_tuple(
                        baseline_model.predict(annualized_daily_fixture, summed=True),
                        reporting_model.predict(annualized_daily_fixture, summed=True))
                    raw_derivatives.append({
                        'series': series,
                        'description': description,
                        'orderable': [None,],
                        'value': [value,],
                        'variance': [variance,]
                    })
                except:
                    _report_failed_derivative(series)

                series = 'Baseline model minus reporting model, normal year'
                description = '''Predicted usage according to the baseline model
                                 over the normal weather year, minus the predicted
                                 usage according to the reporting model over the normal
                                 weather year.'''
                try:
                    value, variance = subtract_value_variance_tuple(
                        baseline_model.predict(annualized_daily_fixture, summed=False),
                        reporting_model.predict(annualized_daily_fixture, summed=False))
                    raw_derivatives.append({
                        'series': series,
                        'description': description,
                        'orderable': [i.isoformat() for i in annualized_daily_fixture.index],
                        'value': value.tolist(),
                        'variance': variance.tolist()
                    })
                except:
                    _report_failed_derivative(series)

            if baseline_model_success:
                if weather_normal_source_success:
                    series = 'Cumulative baseline model, normal year'
                    description = '''Total predicted usage according to the baseline model
                                     over the normal weather year. Days for which normal
                                     year weather data does not exist are removed.'''
                    try:
                        value, variance = baseline_model.predict(
                                    annualized_daily_fixture, summed=True)
                        raw_derivatives.append({
                            'series': series,
                            'description': description,
                            'orderable': [None,],
                            'value': [value,],
                            'variance': [variance,]
                        })
                    except:
                        _report_failed_derivative(series)

                    series = 'Baseline model, normal year'
                    description = '''Predicted usage according to the baseline model
                                     over the normal weather year.'''
                    try:
                        value, variance = baseline_model.predict(
                                    annualized_daily_fixture, summed=False)
                        raw_derivatives.append({
                            'series': series,
                            'description': description,
                            'orderable': [i.isoformat() for i in annualized_daily_fixture.index],
                            'value': value.tolist(),
                            'variance': variance.tolist()
                        })
                    except:
                        _report_failed_derivative(series)

                if weather_source_success and reporting_period_fixture_success:
                    series = 'Cumulative baseline model, reporting period'
                    description = '''Total predicted usage according to the baseline model
                                     over the reporting period. Days for which reporting
                                     period weather data does not exist are removed.'''
                    try:
                        value, variance = baseline_model.predict(
                                reporting_period_daily_fixture, summed=True)
                        raw_derivatives.append({
                            'series': series,
                            'description': description,
                            'orderable': [None,],
                            'value': [value,],
                            'variance': [variance,]
                        })
                    except:
                        _report_failed_derivative(series)

                    series = 'Baseline model, reporting period'
                    description = '''Predicted usage according to the baseline model
                                     over the reporting period.'''
                    try:
                        value, variance = baseline_model.predict(
                                reporting_period_daily_fixture, summed=False)
                        raw_derivatives.append({
                            'series': series,
                            'description': description,
                            'orderable': [i.isoformat() for i in reporting_period_daily_fixture.index],
                            'value': value.tolist(),
                            'variance': variance.tolist()
                        })
                    except:
                        _report_failed_derivative(series)

                    series = 'Masked baseline model, reporting period'
                    description = '''Predicted usage according to the baseline model
                                     over the reporting period, null where values are
                                     missing in either the observed usage or
                                     temperature data.'''

                    try:
                        value, variance = baseline_model.predict(
                                reporting_period_daily_fixture, summed=False)

                        raw_derivatives.append({
                            'series': series,
                            'description': description,
                            'orderable': [i.isoformat() for i in reporting_period_daily_fixture.index],
                            'value': [
                                v if not reporting_mask.get(i, True) else None
                                for i, v in value.iteritems()
                            ],
                            'variance': [
                                v if not reporting_mask.get(i, True) else None
                                for i, v in variance.iteritems()
                            ],
                        })

                    except:
                        _report_failed_derivative(series)

                    series = 'Cumulative baseline model minus observed, reporting period'
                    description = '''Total predicted usage according to the baseline model
                                     minus observed usage over the reporting period.
                                     Days for which reporting period weather data or usage
                                     do not exist are removed.'''
                    try:
                        value, variance = subtract_value_variance_tuple(
                                baseline_model.predict(
                                    reporting_period_daily_fixture, summed=True),
                                (reporting_period_data.sum(), 0)
                            )
                        raw_derivatives.append({
                            'series': series,
                            'description': description,
                            'orderable': [None,],
                            'value': [value,],
                            'variance': [variance,]
                        })
                    except:
                        _report_failed_derivative(series)

                    series = 'Baseline model minus observed, reporting period'
                    description = '''Predicted usage according to the baseline model
                                     minus observed usage over the reporting period.'''
                    try:
                        value, variance = subtract_value_variance_tuple(
                                baseline_model.predict(
                                    reporting_period_daily_fixture, summed=False),
                                (reporting_period_data, 0)
                            )
                        raw_derivatives.append({
                            'series': series,
                            'description': description,
                            'orderable': [i.isoformat() for i in reporting_period_daily_fixture.index],
                            'value': value.tolist(),
                            'variance': variance.tolist()
                        })
                    except:
                        _report_failed_derivative(series)

                    series = 'Masked baseline model minus observed, reporting period'
                    description = '''Predicted usage according to the baseline model
                                     minus observed usage over the reporting period,
                                     null where values are missing in either
                                     the observed usage or temperature data.'''
                    try:
                        value, variance = subtract_value_variance_tuple(
                                baseline_model.predict(
                                    reporting_period_daily_fixture, summed=False),
                                (reporting_period_data, 0)
                            )
                        raw_derivatives.append({
                            'series': series,
                            'description': description,
                            'orderable': [i.isoformat() for i in reporting_period_daily_fixture.index],
                            'value': [
                                v if not reporting_mask.get(i, True) else None
                                for i, v in value.iteritems()
                            ],
                            'variance': [
                                v if not reporting_mask.get(i, True) else None
                                for i, v in variance.iteritems()
                            ],
                        })
                    except:
                        _report_failed_derivative(series)

                if weather_source_success and baseline_period_fixture_success:
                    series = 'Baseline model, baseline period'
                    description = '''Predicted usage according to the baseline model
                                     over the baseline period.'''
                    try:
                        value, variance = baseline_model.predict(
                                baseline_period_daily_fixture, summed=False)
                        raw_derivatives.append({
                            'series': series,
                            'description': description,
                            'orderable': [i.isoformat() for i in baseline_period_daily_fixture.index],
                            'value': value.tolist(),
                            'variance': variance.tolist()
                        })
                    except:
                        _report_failed_derivative(series)

            if reporting_model_success:
                if weather_normal_source_success:
                    series = 'Cumulative reporting model, normal year'
                    description = '''Total predicted usage according to the reporting model
                                     over the reporting period.  Days for which normal year
                                     weather data does not exist are removed.'''
                    try:
                        value, variance = reporting_model.predict(
                                    annualized_daily_fixture, summed=True)
                        raw_derivatives.append({
                            'series': series,
                            'description': description,
                            'orderable': [None,],
                            'value': [value,],
                            'variance': [variance,]
                        })
                    except:
                        _report_failed_derivative(series)

                    series = 'Reporting model, normal year'
                    description = '''Predicted usage according to the reporting model
                                     over the reporting period.'''
                    try:
                        value, variance = reporting_model.predict(
                                    annualized_daily_fixture, summed=False)
                        raw_derivatives.append({
                            'series': series,
                            'description': description,
                            'orderable': [i.isoformat() for i in annualized_daily_fixture.index],
                            'value': value.tolist(),
                            'variance': variance.tolist()
                        })
                    except:
                        _report_failed_derivative(series)

                if weather_source_success and reporting_period_fixture_success:
                    series = 'Reporting model, reporting period'
                    description = '''Predicted usage according to the reporting model
                                     over the reporting period.'''
                    try:
                        value, variance = reporting_model.predict(
                                reporting_period_daily_fixture, summed=False)
                        raw_derivatives.append({
                            'series': series,
                            'description': description,
                            'orderable': [i.isoformat() for i in reporting_period_daily_fixture.index],
                            'value': value.tolist(),
                            'variance': variance.tolist()
                        })
                    except:
                        _report_failed_derivative(series)

            series = 'Cumulative observed, reporting period'
            description = '''Total observed usage over the reporting period.
                             Days for which weather data does not exist
                             are NOT removed.'''
            try:
                raw_derivatives.append({
                    'series': series,
                    'description': description,
                    'orderable': [None],
                    'value': [reporting_period_data.sum(),],
                    'variance': [0,]
                })
            except:
                _report_failed_derivative(series)

            series = 'Observed, reporting period'
            description = '''Observed usage over the reporting period.'''
            try:
                raw_derivatives.append({
                    'series': series,
                    'description': description,
                    'orderable': [i.isoformat() for i in reporting_period_data.index],
                    'value': reporting_period_data.values.tolist(),
                    'variance': [0 for _ in range(reporting_period_data.shape[0])]
                })
            except:
                _report_failed_derivative(series)

            series = 'Masked observed, reporting period'
            description = '''Observed usage over the reporting period,
                             null where values are missing in either
                             the observed usage or temperature data.'''
            try:
                raw_derivatives.append({
                    'series': series,
                    'description': description,
                    'orderable': [i.isoformat() for i in reporting_period_data.index],
                    'value': [
                        v if not reporting_mask.get(i, True) else None
                        for i, v in reporting_period_data.iteritems()
                    ],
                    'variance': [
                        0 if not reporting_mask.get(i, True) else None
                        for i, v in reporting_period_data.iteritems()
                    ],
                })
            except:
                _report_failed_derivative(series)

            series = 'Cumulative observed, baseline period'
            description = '''Total observed usage over the baseline period.
                             Days for which weather data does not exist
                             are NOT removed.'''
            try:
                raw_derivatives.append({
                    'series': series,
                    'description': description,
                    'orderable': [None],
                    'value': [baseline_period_data.sum()],
                    'variance': [0,]
                })
            except:
                _report_failed_derivative(series)

            series = 'Observed, baseline period'
            description = '''Observed usage over the baseline period.'''
            try:
                raw_derivatives.append({
                    'series': series,
                    'description': description,
                    'orderable': [i.isoformat() for i in baseline_period_data.index],
                    'value': baseline_period_data.values.tolist(),
                    'variance': [0 for _ in range(baseline_period_data.shape[0])]
                })
            except:
                _report_failed_derivative(series)

            series = 'Observed, project period'
            description = '''Observed usage over the project period.'''
            try:
                raw_derivatives.append({
                    'series': series,
                    'description': description,
                    'orderable': [i.isoformat() for i in project_period_data.index],
                    'value': project_period_data.values.tolist(),
                    'variance': [0 for _ in range(project_period_data.shape[0])]
                })
            except:
                _report_failed_derivative(series)

            if weather_source_success:
                series = 'Temperature, baseline period'
                description = '''Observed temperature (degF) over the baseline period.'''
                try:
                    raw_derivatives.append({
                        'series': series,
                        'description': description,
                        'orderable': [
                            i.isoformat() for i in unmasked_baseline_period_daily_fixture.index],
                        'value': unmasked_baseline_period_daily_fixture['tempF'].values.tolist(),
                        'variance': [0 for _ in range(unmasked_baseline_period_daily_fixture['tempF'].shape[0])]
                    })
                except:
                    _report_failed_derivative(series)

                series = 'Temperature, reporting period'
                description = '''Observed temperature (degF) over the reporting period.'''
                try:
                    raw_derivatives.append({
                        'series': series,
                        'description': description,
                        'orderable': [
                            i.isoformat() for i in unmasked_reporting_period_daily_fixture.index],
                        'value': unmasked_reporting_period_daily_fixture['tempF'].values.tolist(),
                        'variance': [0 for _ in range(unmasked_reporting_period_daily_fixture['tempF'].shape[0])]
                    })
                except:
                    _report_failed_derivative(series)

                series = 'Masked temperature, reporting period'
                description = '''Observed temperature (degF) over the reporting
                                 period, null where values are missing in either
                                 the observed usage or temperature data.'''
                try:
                    raw_derivatives.append({
                        'series': series,
                        'description': description,
                        'orderable': [
                            i.isoformat() for i in unmasked_reporting_period_daily_fixture.index],
                        'value': [
                            v if not reporting_mask.get(i, True) else None
                            for i, v in unmasked_reporting_period_daily_fixture['tempF'].iteritems()
                        ],
                        'variance': [
                            0 if not reporting_mask.get(i, True) else None
                            for i, v in unmasked_reporting_period_daily_fixture['tempF'].iteritems()
                        ],
                    })
                except:
                    _report_failed_derivative(series)

            if weather_normal_source_success:
                series = 'Temperature, normal year'
                description = '''Observed temperature (degF) over the normal year.'''
                try:
                    raw_derivatives.append({
                        'series': series,
                        'description': description,
                        'orderable': [
                            i.isoformat() for i in annualized_daily_fixture.index],
                        'value': annualized_daily_fixture['tempF'].values.tolist(),
                        'variance': [0 for _ in range(annualized_daily_fixture['tempF'].shape[0])]
                    })
                except:
                    _report_failed_derivative(series)

            series = 'Inclusion mask, baseline period'
            description = '''Mask for baseline period data which is included in
                             model and savings cumulatives.'''
            try:
                raw_derivatives.append({
                    'series': series,
                    'description': description,
                    'orderable': [i.isoformat() for i in baseline_mask.index],
                    'value': [bool(v) for v in baseline_mask.values],
                    'variance': [0 for _ in range(baseline_mask.shape[0])]
                })
            except:
                _report_failed_derivative(series)

            series = 'Inclusion mask, reporting period'
            description = '''Mask for reporting period data which is included in
                             model and savings cumulatives.'''
            try:
                raw_derivatives.append({
                    'series': series,
                    'description': description,
                    'orderable': [i.isoformat() for i in reporting_mask.index],
                    'value': [bool(v) for v in reporting_mask.values],
                    'variance': [0 for _ in range(reporting_mask.shape[0])]
                })
            except:
                _report_failed_derivative(series)

            derivatives += [
                Derivative(
                    (baseline_label, reporting_label),
                    d['series'],
                    reduce(lambda a, b: a + ' ' + b, d['description'].split()),
                    d['orderable'],
                    d['value'],
                    d['variance'],
                )
                for d in raw_derivatives
            ]

        output["derivatives"] = serialize_derivatives(derivatives)
        output["status"] = SUCCESS
        return output
