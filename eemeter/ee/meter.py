import logging
from collections import OrderedDict, namedtuple

from six import string_types
from functools import reduce

from eemeter import get_version
from eemeter.modeling.formatters import (
    ModelDataFormatter,
    ModelDataBillingFormatter,
)
from eemeter.modeling.models import CaltrackMonthlyModel, CaltrackDailyModel, HourlyDayOfWeekModel

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

from eemeter.ee.derivatives import (
    unpack,
    hdd_balance_point_baseline,
    hdd_coefficient_baseline,
    cdd_balance_point_baseline,
    cdd_coefficient_baseline,
    intercept_baseline,
    hdd_balance_point_reporting,
    hdd_coefficient_reporting,
    cdd_balance_point_reporting,
    cdd_coefficient_reporting,
    intercept_reporting,
    cumulative_baseline_model_minus_reporting_model_normal_year,
    baseline_model_minus_reporting_model_normal_year,
    cumulative_baseline_model_normal_year,
    baseline_model_normal_year,
    cumulative_baseline_model_reporting_period,
    baseline_model_reporting_period,
    masked_baseline_model_reporting_period,
    cumulative_baseline_model_minus_observed_reporting_period,
    baseline_model_minus_observed_reporting_period,
    masked_baseline_model_minus_observed_reporting_period,
    baseline_model_baseline_period,
    cumulative_reporting_model_normal_year,
    reporting_model_normal_year,
    reporting_model_reporting_period,
    cumulative_observed_reporting_period,
    observed_reporting_period,
    masked_observed_reporting_period,
    cumulative_observed_baseline_period,
    observed_baseline_period,
    observed_project_period,
    temperature_baseline_period,
    temperature_reporting_period,
    masked_temperature_reporting_period,
    temperature_normal_year,
    baseline_mask,
    reporting_mask,
    normal_year_resource_curve,
    reporting_period_resource_curve,
    normal_year_co2_avoided
)

import pandas as pd

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

    def __init__(self, **kwargs):

        self.kwargs = kwargs
        default_model_mapping = kwargs.get('default_model_mapping', None)
        default_formatter_mapping = kwargs.get('default_formatter_mapping', None)
        weather_station_mapping = kwargs.get('weather_station_mapping', 'default')
        weather_normal_station_mapping = kwargs.get('weather_normal_station_mapping', 'default')

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
            day_of_week_elec_model_daily = (HourlyDayOfWeekModel, {
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

                ('DAY_OF_WEEK_ELECTRICITY_CONSUMPTION_SUPPLIED', None):
                    day_of_week_elec_model_daily,
            }

        self.default_formatter_mapping = default_formatter_mapping
        self.default_model_mapping = default_model_mapping

        if weather_station_mapping not in ['default', 'CZ2010']:
            raise ValueError(
                'weather_station_mapping="{}" not recognized.'
                ' Use "default" or "CZ2010".'
                .format(weather_station_mapping)
            )
        else:
            self.weather_station_mapping = weather_station_mapping

        if weather_normal_station_mapping not in ['default', 'CZ2010']:
            raise ValueError(
                'weather_normal_station_mapping="{}" not recognized.'
                ' Use "default" or "CZ2010".'
                .format(weather_normal_station_mapping)
            )
        else:
            self.weather_normal_station_mapping = weather_normal_station_mapping

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
                    class_name_map = {
                        f.__name__: f
                        for f in [CaltrackMonthlyModel, CaltrackDailyModel, HourlyDayOfWeekModel]
                    }
                    ModelClass = class_name_map[custom_model_class]
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

            ("meter_kwargs", self.kwargs),
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
        use_cz2010 = (self.weather_station_mapping == 'CZ2010')
        if weather_source is None:
            weather_source = get_weather_source(site, use_cz2010=use_cz2010)

            if weather_source is None:
                message = (
                    "Could not find weather normal source matching site {}"
                    .format(site)
                )
                weather_source_usaf_id = None
            else:
                message = "Using weather_source {}".format(weather_source)
                weather_source_usaf_id = weather_source.usaf_id
        else:
            message = "Using supplied weather_source"
            weather_source_usaf_id = weather_source.usaf_id
        output['weather_source_station'] = weather_source_usaf_id
        output['logs'].append(message)
        logger.debug(message)

        if weather_normal_source is None:

            use_cz2010 = (self.weather_normal_station_mapping == 'CZ2010')
            weather_normal_source = get_weather_normal_source(site, use_cz2010=use_cz2010)
            if weather_normal_source is None:
                message = (
                    "Could not find weather normal source matching site {}"
                    .format(site)
                )
                weather_normal_source_usaf_id = None
            else:
                message = (
                    "Using weather_normal_source {}"
                    .format(weather_normal_source)
                )
                weather_normal_source_usaf_id = weather_normal_source.usaf_id
        else:
            message = "Using supplied weather_normal_source"
            weather_normal_source_usaf_id = weather_normal_source.usaf_id
        output['weather_normal_source_station'] = weather_normal_source_usaf_id
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
        derivative_freq = 'D'
        if 'freq_str' in formatter_kwargs.keys() and \
                formatter_kwargs['freq_str'] == 'H':
            derivative_freq = 'H'

        derivatives = []
        for ((baseline_label, reporting_label),
                (baseline_period, reporting_period)) in \
                modeling_period_set.iter_modeling_period_groups():
            raw_derivatives = []
            deriv_input = unpack(modeled_trace, baseline_label, reporting_label,
                                 baseline_period, reporting_period,
                                 weather_source, weather_normal_source,
                                 site, derivative_freq=derivative_freq)
            if deriv_input is None:
                continue
            raw_derivatives.extend([
                hdd_balance_point_baseline(deriv_input),
                hdd_coefficient_baseline(deriv_input),
                cdd_balance_point_baseline(deriv_input),
                cdd_coefficient_baseline(deriv_input),
                intercept_baseline(deriv_input),
                hdd_balance_point_reporting(deriv_input),
                hdd_coefficient_reporting(deriv_input),
                cdd_balance_point_reporting(deriv_input),
                cdd_coefficient_reporting(deriv_input),
                intercept_reporting(deriv_input),
                cumulative_baseline_model_minus_reporting_model_normal_year(deriv_input),
                baseline_model_minus_reporting_model_normal_year(deriv_input),
                cumulative_baseline_model_normal_year(deriv_input),
                baseline_model_normal_year(deriv_input),
                cumulative_baseline_model_reporting_period(deriv_input),
                baseline_model_reporting_period(deriv_input),
                masked_baseline_model_reporting_period(deriv_input),
                cumulative_baseline_model_minus_observed_reporting_period(deriv_input),
                baseline_model_minus_observed_reporting_period(deriv_input),
                masked_baseline_model_minus_observed_reporting_period(deriv_input),
                baseline_model_baseline_period(deriv_input),
                cumulative_reporting_model_normal_year(deriv_input),
                reporting_model_normal_year(deriv_input),
                reporting_model_reporting_period(deriv_input),
                cumulative_observed_reporting_period(deriv_input),
                observed_reporting_period(deriv_input),
                masked_observed_reporting_period(deriv_input),
                cumulative_observed_baseline_period(deriv_input),
                observed_baseline_period(deriv_input),
                observed_project_period(deriv_input),
                temperature_baseline_period(deriv_input),
                temperature_reporting_period(deriv_input),
                masked_temperature_reporting_period(deriv_input),
                temperature_normal_year(deriv_input),
                baseline_mask(deriv_input),
                reporting_mask(deriv_input),
                reporting_period_resource_curve(deriv_input)
            ])

            resource_curve_normal_year = normal_year_resource_curve(deriv_input)
            raw_derivatives.extend([resource_curve_normal_year])

            if resource_curve_normal_year is not None:
                resource_curve_normal_year = pd.Series(
                    resource_curve_normal_year['value'],
                    index=pd.to_datetime(resource_curve_normal_year['orderable']))
                raw_derivatives.extend([normal_year_co2_avoided(
                    deriv_input, resource_curve_normal_year)])

            derivatives += [
                Derivative(
                    (baseline_label, reporting_label),
                    d['series'],
                    reduce(lambda a, b: a + ' ' + b, d['description'].split()),
                    d['orderable'],
                    d['value'],
                    d['variance'],
                )
                for d in raw_derivatives if d is not None
            ]

        output["derivatives"] = serialize_derivatives(derivatives)
        output["status"] = SUCCESS
        return output
