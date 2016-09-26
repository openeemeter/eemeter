import logging
from collections import OrderedDict

from eemeter import get_version
from eemeter.ee.derivatives import (
    DerivativePair,
    Derivative,
    annualized_weather_normal,
    gross_predicted,
)
from eemeter.modeling.formatters import (
    ModelDataBillingFormatter,
    ModelDataFormatter,
)
from eemeter.modeling.models import (
    BillingElasticNetCVModel,
    SeasonalElasticNetCVModel,
)
from eemeter.modeling.split import (
    SplitModeledEnergyTrace
)
from eemeter.io.serializers import (
    deserialize_meter_input,
    serialize_derivative_pairs,
    serialize_split_modeled_energy_trace,
)
from eemeter.processors.dispatchers import (
    get_approximate_frequency,
    get_energy_modeling_dispatches,
)
from eemeter.processors.interventions import get_modeling_period_set
from eemeter.processors.location import (
    get_weather_normal_source,
    get_weather_source,
)
from eemeter.structures import ZIPCodeSite

logger = logging.getLogger(__name__)


class EnergyEfficiencyMeter(object):
    ''' The standard way of calculating energy efficiency savings values from
    project data.

    Parameters
    ----------
    settings : dict
        Dictionary of settings (ignored; for now, this is a placeholder).
    '''

    def __init__(self, settings=None):
        if settings is None:
            self.settings = {}
        self.settings = settings

    def evaluate(self, project, weather_source=None,
                 weather_normal_source=None):
        ''' Main entry point to the meter, taking in project data and returning
        results indicating energy efficiency performance.

        Parameters
        ----------
        project : eemeter.structures.Project
            Project for which energy effienciency performance is to be
            evaluated.
        weather_source : eemeter.weather.WeatherSource
            Weather source to be used for this meter. Overrides weather source
            found using :code:`project.site`. Useful for test mocking.
        weather_normal_source : eemeter.weather.WeatherSource
            Weather normal source to be used for this meter. Overrides weather
            source found using :code:`project.site`. Useful for test mocking.

        Returns
        -------
        out : dict
            Results of energy efficiency evaluation, organized into the
            following items.

            - :code:`"modeling_period_set"`:
              :code:`eemeter.structures.ModelingPeriodSet` determined from this
              project.
            - :code:`"modeled_energy_traces"`: dict of dispatched modeled
              energy traces.
            - :code:`"modeled_energy_trace_derivatives"`: derivatives for each
              modeled energy trace.
            - :code:`"project_derivatives"`: Project summaries for derivatives.
            - :code:`"weather_source"`: Matched weather source
            - :code:`"weather_normal_source"`: Matched weather normal source.
        '''

        modeling_period_set = get_modeling_period_set(project.interventions)

        if weather_source is None:
            weather_source = get_weather_source(project.site)
        else:
            logger.info("Using supplied weather_source")

        if weather_normal_source is None:
            weather_normal_source = get_weather_normal_source(project.site)
        else:
            logger.info("Using supplied weather_normal_source")

        dispatches = get_energy_modeling_dispatches(
            modeling_period_set, project.energy_trace_set)

        derivatives = {}
        for trace_label, modeled_energy_trace in dispatches.items():

            trace_derivatives = {}
            derivatives[trace_label] = trace_derivatives

            if modeled_energy_trace is None:
                continue

            modeled_energy_trace.fit(weather_source)

            for group_label, (_, reporting_period) in \
                    modeling_period_set.iter_modeling_period_groups():

                period_derivatives = {
                    "BASELINE": {},
                    "REPORTING": {},
                }
                trace_derivatives[group_label] = \
                    period_derivatives

                baseline_label, reporting_label = group_label

                baseline_output = modeled_energy_trace.fit_outputs[
                    baseline_label]
                reporting_output = modeled_energy_trace.fit_outputs[
                    reporting_label]

                if baseline_output["status"] == "SUCCESS":
                    awn = modeled_energy_trace.compute_derivative(
                        baseline_label,
                        annualized_weather_normal,
                        {
                            "weather_normal_source": weather_normal_source,
                        })
                    if awn is not None:
                        period_derivatives["BASELINE"].update(awn)

                    gp = modeled_energy_trace.compute_derivative(
                        baseline_label,
                        gross_predicted,
                        {
                            "weather_source": weather_source,
                            "reporting_period": reporting_period,
                        })
                    if gp is not None:
                        period_derivatives["BASELINE"].update(gp)

                if reporting_output["status"] == "SUCCESS":
                    awn = modeled_energy_trace.compute_derivative(
                        reporting_label,
                        annualized_weather_normal,
                        {
                            "weather_normal_source": weather_normal_source,
                        })
                    if awn is not None:
                        period_derivatives["REPORTING"].update(awn)

                    gp = modeled_energy_trace.compute_derivative(
                        reporting_label,
                        gross_predicted,
                        {
                            "weather_source": weather_source,
                            "reporting_period": reporting_period,
                        })
                    if gp is not None:
                        period_derivatives["REPORTING"].update(gp)

        project_derivatives = self._get_project_derivatives(
            modeling_period_set,
            project.energy_trace_set,
            derivatives)

        return {
            "modeling_period_set": modeling_period_set,
            "modeled_energy_traces": dispatches,
            "modeled_energy_trace_derivatives": derivatives,
            "project_derivatives": project_derivatives,
            "weather_source": weather_source,
            "weather_normal_source": weather_normal_source,
        }

    def _get_project_derivatives(self, modeling_period_set, energy_trace_set,
                                 derivatives):

        # create list of project derivative labels

        target_trace_interpretations = [
            {
                'name': 'ELECTRICITY_CONSUMPTION_SUPPLIED',
                'interpretations': (
                    'ELECTRICITY_CONSUMPTION_SUPPLIED',
                ),
                'target_unit': 'KWH',
                'requirements': ['BASELINE', 'REPORTING'],
            },
            {
                'name': 'NATURAL_GAS_CONSUMPTION_SUPPLIED',
                'interpretations': (
                    'NATURAL_GAS_CONSUMPTION_SUPPLIED',
                ),
                'target_unit': 'KWH',
                'requirements': ['BASELINE', 'REPORTING'],
            },
            {
                'name': 'ALL_FUELS_CONSUMPTION_SUPPLIED',
                'interpretations': (
                    'ELECTRICITY_CONSUMPTION_SUPPLIED',
                    'NATURAL_GAS_CONSUMPTION_SUPPLIED',
                ),
                'target_unit': 'KWH',
                'requirements': ['BASELINE', 'REPORTING'],
            },
            {
                'name': 'ELECTRICITY_ON_SITE_GENERATION_UNCONSUMED',
                'interpretations': (
                    'ELECTRICITY_ON_SITE_GENERATION_UNCONSUMED',
                ),
                'target_unit': 'KWH',
                'requirements': ['REPORTING'],
            },
        ]

        target_outputs = [
            ('annualized_weather_normal', 'ANNUALIZED_WEATHER_NORMAL'),
            ('gross_predicted', 'GROSS_PREDICTED'),
        ]

        def _get_target_output(trace_label, modeling_period_group_label,
                               output_key):
            trace_output = derivatives.get(trace_label, None)
            if trace_output is None:
                return None, None

            group_output = trace_output.get(modeling_period_group_label, None)
            if group_output is None:
                return None, None

            baseline_output = group_output['BASELINE']
            reporting_output = group_output['REPORTING']

            baseline = baseline_output.get(output_key, None)
            reporting = reporting_output.get(output_key, None)
            return baseline, reporting

        project_derivatives = {}

        # for each modeling period group
        for group_label, _ in \
                modeling_period_set.iter_modeling_period_groups():

            group_derivatives = {}
            project_derivatives[group_label] = group_derivatives

            # create the group derivatives
            for spec in target_trace_interpretations:
                name = spec["name"]
                interpretations = spec["interpretations"]
                target_unit = spec["target_unit"]
                requirements = spec["requirements"]

                if name not in group_derivatives:
                    group_derivatives[name] = None

                for trace_label, trace in energy_trace_set.itertraces():

                    if trace.interpretation not in interpretations:
                        continue

                    for output_key, output_label in target_outputs:

                        baseline_output, reporting_output = \
                            _get_target_output(
                                trace_label, group_label, output_key)

                        if (('BASELINE' in requirements and
                             baseline_output is None) or
                            ('REPORTING' in requirements and
                             reporting_output is None)):
                            continue

                        if baseline_output is None:
                            baseline_output = (0.0, 0.0, 0.0, 0)
                        else:
                            baseline_output = baseline_output[:4]

                        if reporting_output is None:
                            reporting_output = (0.0, 0.0, 0.0, 0)
                        else:
                            reporting_output = reporting_output[:4]

                        baseline_output = _change_units(
                            baseline_output, trace.unit, target_unit)
                        reporting_output = _change_units(
                            reporting_output, trace.unit, target_unit)

                        if group_derivatives[name] is None:
                            group_derivatives[name] = {
                                'BASELINE': {
                                    output_key: baseline_output,
                                },
                                'REPORTING': {
                                    output_key: reporting_output,
                                },
                                'unit': target_unit,
                            }
                        else:
                            old_baseline_output = \
                                group_derivatives[name]['BASELINE'].get(
                                    output_key, None)
                            old_reporting_output = \
                                group_derivatives[name]['REPORTING'].get(
                                    output_key, None)

                            if old_baseline_output is None:
                                group_derivatives[name]['BASELINE'][
                                    output_key] = baseline_output
                            else:
                                group_derivatives[name]['BASELINE'][
                                    output_key] = _add_errors(
                                        baseline_output,
                                        old_baseline_output)

                            if old_reporting_output is None:
                                group_derivatives[name]['REPORTING'][
                                    output_key] = reporting_output
                            else:
                                group_derivatives[name]['REPORTING'][
                                    output_key] = _add_errors(
                                        reporting_output,
                                        old_reporting_output)
        return project_derivatives


def _add_errors(errors1, errors2):
    # TODO add autocorrelation correction
    mean1, lower1, upper1, n1 = errors1
    mean2, lower2, upper2, n2 = errors2

    mean = mean1 + mean2
    lower = (lower1**2 + lower2**2)**0.5
    upper = (upper1**2 + upper2**2)**0.5
    n = n1 + n2
    return (mean, lower, upper, n)


def _change_units(errors, units_from, units_to):

    factor = None

    if units_from == "KWH":

        if units_to == "KWH":
            factor = 1.0
        elif units_to == "THERM":
            factor = 0.0341296

    elif units_from == "THERM":

        if units_to == "KWH":
            factor = 29.3001
        elif units_to == "THERM":
            factor = 1.0

    # shouldn't fail - all units should either be KWH or THERM
    assert factor is not None

    mean, upper, lower, n = errors
    return (mean*factor, upper*factor, lower*factor, n)


class EnergyEfficiencyMeterTraceCentric(object):
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
                ('NATURAL_GAS_CONSUMPTION_SUPPLIED', '15T'): daily_formatter,
                ('ELECTRICITY_CONSUMPTION_SUPPLIED', '15T'): daily_formatter,
                ('ELECTRICITY_ON_SITE_GENERATION_UNCONSUMED', '15T'):
                    daily_formatter,

                ('NATURAL_GAS_CONSUMPTION_SUPPLIED', '30T'): daily_formatter,
                ('ELECTRICITY_CONSUMPTION_SUPPLIED', '30T'): daily_formatter,
                ('ELECTRICITY_ON_SITE_GENERATION_UNCONSUMED', '30T'):
                    daily_formatter,

                ('NATURAL_GAS_CONSUMPTION_SUPPLIED', 'H'): daily_formatter,
                ('ELECTRICITY_CONSUMPTION_SUPPLIED', 'H'): daily_formatter,
                ('ELECTRICITY_ON_SITE_GENERATION_UNCONSUMED', 'H'):
                    daily_formatter,

                ('NATURAL_GAS_CONSUMPTION_SUPPLIED', 'D'): daily_formatter,
                ('ELECTRICITY_CONSUMPTION_SUPPLIED', 'D'): daily_formatter,
                ('ELECTRICITY_ON_SITE_GENERATION_UNCONSUMED', 'D'):
                    daily_formatter,

                ('NATURAL_GAS_CONSUMPTION_SUPPLIED', None): billing_formatter,
                ('ELECTRICITY_CONSUMPTION_SUPPLIED', None): billing_formatter,
            }

        if default_model_mapping is None:
            seasonal_model = (SeasonalElasticNetCVModel, {
                'cooling_base_temp': 65,
                'heating_base_temp': 65,
            })
            billing_model = (BillingElasticNetCVModel, {
                'cooling_base_temp': 65,
                'heating_base_temp': 65,
            })
            default_model_mapping = {
                ('NATURAL_GAS_CONSUMPTION_SUPPLIED', '15T'): seasonal_model,
                ('ELECTRICITY_CONSUMPTION_SUPPLIED', '15T'): seasonal_model,
                ('ELECTRICITY_ON_SITE_GENERATION_UNCONSUMED', '15T'):
                    seasonal_model,

                ('NATURAL_GAS_CONSUMPTION_SUPPLIED', '30T'): seasonal_model,
                ('ELECTRICITY_CONSUMPTION_SUPPLIED', '30T'): seasonal_model,
                ('ELECTRICITY_ON_SITE_GENERATION_UNCONSUMED', '30T'):
                    seasonal_model,

                ('NATURAL_GAS_CONSUMPTION_SUPPLIED', 'H'): seasonal_model,
                ('ELECTRICITY_CONSUMPTION_SUPPLIED', 'H'): seasonal_model,
                ('ELECTRICITY_ON_SITE_GENERATION_UNCONSUMED', 'H'):
                    seasonal_model,

                ('NATURAL_GAS_CONSUMPTION_SUPPLIED', 'D'): seasonal_model,
                ('ELECTRICITY_CONSUMPTION_SUPPLIED', 'D'): seasonal_model,
                ('ELECTRICITY_ON_SITE_GENERATION_UNCONSUMED', 'D'):
                    seasonal_model,

                ('NATURAL_GAS_CONSUMPTION_SUPPLIED', None): billing_model,
                ('ELECTRICITY_CONSUMPTION_SUPPLIED', None): billing_model,
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
        else:
            message = "Using supplied weather_source"
            logger.info(message)
        output['weather_source_station'] = weather_source.station
        output['logs'].append(message)

        if weather_normal_source is None:
            weather_normal_source = get_weather_normal_source(site)
            message = "Using weather_normal_source {}".format(weather_source)
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
            modeling_period_label: ModelClass(**model_kwargs)
            for modeling_period_label, _ in
            modeling_period_set.iter_modeling_periods()
        }

        modeled_energy_trace = SplitModeledEnergyTrace(
            trace, formatter_instance, model_mapping, modeling_period_set)

        modeled_energy_trace.fit(weather_source)
        output["modeled_energy_trace"] = \
            serialize_split_modeled_energy_trace(modeled_energy_trace)

        # Step 9: for each modeling period group, create derivatives
        derivative_pairs = []
        for group_label, (_, reporting_period) in \
                modeling_period_set.iter_modeling_period_groups():

            baseline_label, reporting_label = group_label

            baseline_output = modeled_energy_trace \
                .fit_outputs[baseline_label]
            reporting_output = modeled_energy_trace \
                .fit_outputs[reporting_label]

            baseline_model_success = (baseline_output["status"] == "SUCCESS")
            reporting_model_success = (reporting_output["status"] == "SUCCESS")

            def _compute_derivative(baseline_label, reporting_label,
                                    interpretation, derivative_func, kwargs):

                baseline_derivative = \
                    Derivative(baseline_label, None, None, None, None, None)
                if baseline_model_success:
                    baseline_derivative = modeled_energy_trace \
                        .compute_derivative(
                            baseline_label, derivative_func, kwargs)
                    if baseline_derivative is not None:
                        value, lower, upper, n, serialized_demand_fixture = \
                            baseline_derivative[interpretation]
                        baseline_derivative = Derivative(
                            baseline_label, value, lower, upper, n,
                            serialized_demand_fixture
                        )

                reporting_derivative = \
                    Derivative(reporting_label, None, None, None, None, None)
                if reporting_model_success:
                    reporting_derivative = modeled_energy_trace \
                        .compute_derivative(
                            reporting_label, derivative_func, kwargs)
                    if reporting_derivative is not None:
                        value, lower, upper, n, serialized_demand_fixture = \
                            reporting_derivative[interpretation]
                        reporting_derivative = Derivative(
                            reporting_label, value, lower, upper, n,
                            serialized_demand_fixture
                        )

                derivative_pair = DerivativePair(
                    None, interpretation, trace.interpretation, trace.unit,
                    baseline_derivative, reporting_derivative
                )
                return derivative_pair

            derivative_pairs.extend([
                _compute_derivative(
                    baseline_label, reporting_label,
                    "annualized_weather_normal", annualized_weather_normal,
                    {
                        "weather_normal_source": weather_normal_source,
                    }),
                _compute_derivative(
                    baseline_label, reporting_label,
                    "gross_predicted", gross_predicted,
                    {
                        "weather_source": weather_source,
                        "reporting_period": reporting_period,
                    }),
                # more derivatives can go here
            ])

        output["derivatives"] = serialize_derivative_pairs(derivative_pairs)
        output["status"] = SUCCESS
        return output
