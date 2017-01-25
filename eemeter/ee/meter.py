import logging
from collections import OrderedDict

from eemeter import get_version
from eemeter.ee.derivatives import (
    DerivativePair,
    Derivative,
    annualized_weather_normal,
    gross_predicted,
    gross_actual
)
from eemeter.modeling.formatters import (
    CaltrackFormatter,
)
from eemeter.modeling.models import (
    CaltrackModel,
)
from eemeter.modeling.split import (
    SplitModeledEnergyTrace
)
from eemeter.io.serializers import (
    deserialize_meter_input,
    serialize_derivatives,
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
            caltrack_formatter = (CaltrackFormatter, {'grid_search': True})
            default_formatter_mapping = {
                ('NATURAL_GAS_CONSUMPTION_SUPPLIED', '15T'):
                    caltrack_formatter,
                ('ELECTRICITY_CONSUMPTION_SUPPLIED', '15T'):
                    caltrack_formatter,
                ('ELECTRICITY_ON_SITE_GENERATION_UNCONSUMED', '15T'):
                    caltrack_formatter,

                ('NATURAL_GAS_CONSUMPTION_SUPPLIED', '30T'):
                    caltrack_formatter,
                ('ELECTRICITY_CONSUMPTION_SUPPLIED', '30T'):
                    caltrack_formatter,
                ('ELECTRICITY_ON_SITE_GENERATION_UNCONSUMED', '30T'):
                    caltrack_formatter,

                ('NATURAL_GAS_CONSUMPTION_SUPPLIED', 'H'):
                    caltrack_formatter,
                ('ELECTRICITY_CONSUMPTION_SUPPLIED', 'H'):
                    caltrack_formatter,
                ('ELECTRICITY_ON_SITE_GENERATION_UNCONSUMED', 'H'):
                    caltrack_formatter,

                ('NATURAL_GAS_CONSUMPTION_SUPPLIED', 'D'):
                    caltrack_formatter,
                ('ELECTRICITY_CONSUMPTION_SUPPLIED', 'D'):
                    caltrack_formatter,
                ('ELECTRICITY_ON_SITE_GENERATION_UNCONSUMED', 'D'):
                    caltrack_formatter,

                ('NATURAL_GAS_CONSUMPTION_SUPPLIED', None):
                    caltrack_formatter,
                ('ELECTRICITY_CONSUMPTION_SUPPLIED', None):
                    caltrack_formatter,
            }

        if default_model_mapping is None:
            caltrack_gas_model = (CaltrackModel, {
                'fit_cdd': False,
            })
            caltrack_elec_model = (CaltrackModel, {
                'fit_cdd': True,
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
            modeling_period_label: ModelClass(**model_kwargs)
            for modeling_period_label, _ in
            modeling_period_set.iter_modeling_periods()
        }

        modeled_trace = SplitModeledEnergyTrace(
            trace, formatter_instance, model_mapping, modeling_period_set)

        modeled_trace.fit(weather_source)
        output["modeled_energy_trace"] = \
            serialize_split_modeled_energy_trace(modeled_trace)

        # Step 9: for each modeling period group, create derivatives
        derivative_pairs = []
        for (baseline_label, reporting_label), (baseline_period, reporting_period) in \
                modeling_period_set.iter_modeling_period_groups():

            baseline_output = modeled_trace.fit_outputs[baseline_label]
            reporting_output = modeled_trace.fit_outputs[reporting_label]

            baseline_model_success = (baseline_output["status"] == "SUCCESS")
            reporting_model_success = (reporting_output["status"] == "SUCCESS")

            # how modeled_energy_trace.compute_derivative
            baseline_model = modeled_trace.model_mapping[baseline_label]
            reporting_model = modeled_trace.model_mapping[reporting_label]
            formatter = modeled_trace.formatter
            unit = modeled_trace.unit
            baseline_period_star

            def annualized_weather_normal(model):

                normal_index = pd.date_range(
                    '2015-01-01', freq='D', periods=365, tz=pytz.UTC)

                demand_fixture_data = formatter.create_demand_fixture(
                    normal_index, weather_normal_source)

                serialized_demand_fixture = formatter.serialize_demand_fixture(
                    demand_fixture_data)

                value, variance = model.predict(demand_fixture_data, summed=True)
                return value, variance, serialized_demand_fixture

            # TODO create an observed model

            def reporting_cumulative(model, data):
                # TODO
                return value, variance, serialized_demand_fixture

            def reporting_monthly(model, data):
                # TODO
                return []

            def baseline_monthly(model, data):
                # TODO
                return []

            def project_monthly(model, data):
                # TODO
                return []

            # first create basic set of differential derivatives
            def baseline_model_annualized_weather_normal():
                value, variance, serialized_demand_fixture = \
                    annualized_weather_normal(baseline_model)
                return [(None, value, variance, unit, serialized_demand_fixture)]

            def baseline_model_reporting_cumulative():
                value, variance, serialized_demand_fixture = \
                    reporting_cumulative(baseline_model)
                return [(None, value, variance, unit, serialized_demand_fixture)]

            def baseline_model_reporting_monthly():
                return [
                    (orderable, value, variance, unit, serialized_demand_fixture)
                    for orderable, value, variance, unit, serialized_demand_fixture in reporting_monthly(baseline_model)
                ]

            def reporting_model_annualized_weather_normal():
                value, variance, serialized_demand_fixture = \
                    annualized_weather_normal(reporting_model)
                return [(None, value, variance, unit, serialized_demand_fixture)]

            def observed_reporting_cumulative():
                value, variance, serialized_demand_fixture = \
                    reporting_cumulative(observed_model)
                return [(None, value, variance, unit, serialized_demand_fixture)]

            def observed_reporting_monthly():
                return [
                    (orderable, value, variance, unit, serialized_demand_fixture)
                    for orderable, value, variance, unit, serialized_demand_fixture in reporting_monthly(observed_model)
                ]

            def observed_baseline_monthly():
                return [
                    (orderable, value, variance, unit, serialized_demand_fixture)
                    for orderable, value, variance, unit, serialized_demand_fixture in baseline_monthly(observed_model)
                ]

            def observed_project_monthly():
                return [
                    (orderable, value, variance, unit, serialized_demand_fixture)
                    for orderable, value, variance, unit, serialized_demand_fixture in project_monthly(observed_model)
                ]

            derivative_spec = OrderedDict([
                (
                    ('baseline_model', 'annualized_weather_normal'),
                    baseline_model_annualized_weather_normal,
                ), (
                    ('baseline_model', 'reporting_cumulative'),
                    baseline_model_reporting_cumulative
                ), (
                    ('baseline_model', 'reporting_monthly'),
                    reporting_model_annualized_weather_normal,
                ), (
                    ('reporting_model', 'annualized_weather_normal'),
                    reporting_model_annualized_weather_normal
                ), (
                    ('observed', 'reporting_cumulative'),
                    observed_reporting_cumulative
                ),
                    ('observed', 'reporting_monthly'),
                    observed_reporting_monthly
                ), (
                    ('observed', 'baseline_monthly'),
                    observed_baseline_monthly
                ), (  # between baseline and reporting period
                    ('observed', 'project_monthly'),
                    observed_project_monthly
                ), (
            ]

            # >>> derivative_callable(formatter, model, **kwargs)

            def _build_basic_derivatives():
                all_derivatives = []
                for (source, series), func in derivative_spec.items():
                    derivatives = [
                        Derivative(source, series, orderable, value, variance, unit, serialized_demand_fixture)
                        for orderable, value, variance, unit, serialized_demand_fixture in func()
                    ]

                    all_derivatives.extend(derivatives)
                return all_derivatives

            derivatives = _build_derivatives()

            # then create set of differential derivatives
            differential_spec = [
                [('baseline_model', 'annualized_weather_normal'), ('reporting_model')]
            ]


        output["derivatives"] = serialize_derivatives(derivatives)
        output["status"] = SUCCESS
        return output
