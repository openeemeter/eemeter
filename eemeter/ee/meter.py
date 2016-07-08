from collections import defaultdict

from eemeter.processors.collector import LogCollector
from eemeter.processors.interventions import get_modeling_period_set
from eemeter.processors.location import (
    get_weather_source,
    get_weather_normal_source,
)
from eemeter.processors.dispatchers import get_energy_modeling_dispatches
from eemeter.ee.derivatives import annualized_weather_normal


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

            - :code:`"energy_trace_labels"`: labels for energy traces.
            - :code:`"energy_trace_interpretations"`: dict of interpretations
              of energy traces, organized by energy trace label.
            - :code:`"modeling_period_labels"`: labels for modeling periods.
            - :code:`"modeling_period_interpretations"`: dict of
              interpretations of modeling_periods, organized by modeling
              period label.
            - :code:`"modeling_period_groups"`: list of modeling period groups,
              organized into tuples containing modeling period labels.
            - :code:`"modeled_energy_trace_selectors"`: list of selectors for
              modeled energy traces consisting of a tuple of the form
              :code:`(modeling_period_label, energy_trace_label)`
            - :code:`"modeled_energy_traces"`: dict of results keyed by
              :code:`"modeled_energy_trace_selectors"`. Each result contains
              the following items:

                - :code:`"status"`: :code:`"SUCCESS"` or :code:`"FAILURE"`
                - :code:`"annualized_weather_normal"`: output from
                  annualized_weather_normal derivative.
                - :code:`"n"`: number of samples in fit.
                - :code:`"r2"`: R-squared value for model fit.
                - :code:`"rmse"`: Root Mean Square Error of model fit.
                - :code:`"cvrmse"`: Coefficient of Variation of Root Mean
                  Square Error (a normalized version of RMSE).
                - :code:`"model_params"`: Parameters of the model.

            - :code:`"logs"`: Logs collected during meter run.
        '''

        log_collector = LogCollector()

        with log_collector.collect_logs("get_modeling_period_set") as logger:
            modeling_period_set = get_modeling_period_set(
                logger, project.interventions)

        with log_collector.collect_logs("get_weather_source") as logger:
            if weather_source is None:
                weather_source = get_weather_source(logger, project)
            else:
                logger.info("Using supplied weather_source")

        with log_collector.collect_logs("get_weather_normal_source") as logger:
            if weather_normal_source is None:
                weather_normal_source = get_weather_normal_source(
                    logger, project)
            else:
                logger.info("Using supplied weather_normal_source")

        with log_collector.collect_logs(
                "get_energy_modeling_dispatches") as logger:
            dispatches = get_energy_modeling_dispatches(
                logger, modeling_period_set, project.energy_trace_set)

        with log_collector.collect_logs("handle_dispatches") as logger:

            dispatch_outputs = {}
            for key, dispatch in dispatches.items():

                dispatch_outputs[key] = {}

                formatter = dispatch["formatter"]
                model = dispatch["model"]
                filtered_trace = dispatch["filtered_trace"]

                if (formatter is None or model is None or
                        filtered_trace is None):
                    logger.info(
                        'Modeling skipped for "{}" because of empty'
                        ' dispatch:\n'
                        '  formatter={}\n'
                        '  model={}\n'
                        '  filtered_trace={}'
                        .format(key, formatter, model, filtered_trace)
                    )
                    continue

                input_data = formatter.create_input(
                    filtered_trace, weather_source)

                try:
                    output = model.fit(input_data)
                except:
                    logger.warn(
                        'For "{}", {} was not able to fit using input data: {}'
                        .format(key, model, input_data)
                    )
                    dispatch_outputs[key]["status"] = "FAILURE"
                else:
                    logger.info(
                        'Successfully fitted {} to formatted input data for'
                        ' {}.'
                        .format(model, key)
                    )
                    dispatch_outputs[key].update(output)

                    annualized = annualized_weather_normal(
                        formatter, model, weather_normal_source)
                    dispatch_outputs[key].update(annualized)
                    dispatch_outputs[key]["status"] = "SUCCESS"

        project_derivatives = self._get_project_derivatives(
            modeling_period_set,
            project.energy_trace_set,
            dispatches,
            dispatch_outputs)

        output_data = {
            "energy_trace_labels": [
                label
                for label, _ in project.energy_trace_set.itertraces()
            ],
            "energy_trace_interpretations": {
                label: trace.interpretation
                for label, trace in project.energy_trace_set.itertraces()
            },
            "modeling_period_labels": [
                label
                for label, _ in modeling_period_set.get_modeling_periods()
            ],
            "modeling_period_interpretations": {
                label: modeling_period.interpretation
                for label, modeling_period in
                    modeling_period_set.get_modeling_periods()
            },
            "modeling_period_groups": [
                (baseline_label, reporting_label)
                for (baseline_label, _), (reporting_label, _) in
                modeling_period_set.get_modeling_period_groups()
            ],
            "modeled_energy_trace_selectors": list(dispatches.keys()),
            "modeled_energy_traces": dispatch_outputs,
            "logs": log_collector.items
        }

        for key, value in project_derivatives.items():
            output_data.update({key: value})

        return output_data

    def _get_project_derivatives(self, modeling_period_set, energy_trace_set,
                                 dispatches, dispatch_outputs):

        project_derivatives = defaultdict(lambda: (0, 0, 0, 0))

        for baseline, reporting \
                in modeling_period_set.get_modeling_period_groups():
            baseline_label, baseline_period = baseline
            reporting_label, reporting_period = reporting
            for trace_label, trace in energy_trace_set.itertraces():

                # baseline model
                def _get_baseline(label):
                    baseline_model_dispatch_outputs = \
                        dispatch_outputs.get((baseline_label, trace_label))
                    if baseline_model_dispatch_outputs is not None:
                        return baseline_model_dispatch_outputs.get(label)

                # reporting model
                def _get_reporting(label):
                    reporting_model_dispatch_outputs = \
                        dispatch_outputs.get((reporting_label, trace_label))
                    if reporting_model_dispatch_outputs is not None:
                        return reporting_model_dispatch_outputs.get(label)

                def _add_errors(errors1, errors2):
                    # TODO add autocorrelation correction
                    mean1, lower1, upper1, n1 = errors1
                    mean2, lower2, upper2, n2 = errors2

                    mean = mean1 + mean2
                    lower = (lower1**2 + lower2**2)**0.5
                    upper = (upper1**2 + upper2**2)**0.5
                    n = n1 + n2
                    return (mean, lower, upper, n)

                baseline_annualized = _get_baseline(
                    "annualized_weather_normal")
                reporting_annualized = _get_reporting(
                    "annualized_weather_normal")

                both_available = (
                    baseline_annualized is not None and
                    reporting_annualized is not None
                )

                has_valid_baseline_electricity_consumption = (
                    both_available and
                    trace.interpretation == "ELECTRICITY_CONSUMPTION_SUPPLIED"
                )
                has_valid_reporting_electricity_consumption = (
                    both_available and
                    trace.interpretation == "ELECTRICITY_CONSUMPTION_SUPPLIED"
                )
                has_valid_baseline_natural_gas_consumption = (
                    both_available and
                    trace.interpretation == "NATURAL_GAS_CONSUMPTION_SUPPLIED"
                )
                has_valid_reporting_natural_gas_consumption = (
                    both_available and
                    trace.interpretation == "NATURAL_GAS_CONSUMPTION_SUPPLIED"
                )
                has_valid_baseline_generation = (
                    baseline_annualized is not None and
                    trace.interpretation ==
                    "ELECTRICITY_ON_SITE_GENERATION_UNCONSUMED"
                )
                has_valid_reporting_generation = (
                    reporting_annualized is not None and
                    trace.interpretation ==
                    "ELECTRICITY_ON_SITE_GENERATION_UNCONSUMED"
                )

                def update_derivative_column(name, data):
                    project_derivatives[name] = \
                        _add_errors(project_derivatives[name], data)

                fuel_baseline_col = \
                    "total_baseline_normal_annual_fuel_consumption_kWh"
                fuel_reporting_col = \
                    "total_reporting_normal_annual_fuel_consumption_kWh"

                elec_baseline_col = (
                    "total_baseline_normal_annual"
                    "_electricity_consumption_kWh"
                )
                elec_reporting_col = (
                    "total_reporting_normal_annual"
                    "_electricity_consumption_kWh"
                )

                gas_baseline_col = (
                    "total_baseline_normal_annual"
                    "_natural_gas_consumption_therms"
                )
                gas_reporting_col = (
                    "total_reporting_normal_annual"
                    "_natural_gas_consumption_therms"
                )

                solar_baseline_col = \
                    "total_baseline_normal_annual_solar_generation_kWh"
                solar_reporting_col = \
                    "total_reporting_normal_annual_solar_generation_kWh"

                if has_valid_baseline_electricity_consumption:
                    update_derivative_column(fuel_baseline_col,
                                             baseline_annualized)
                    update_derivative_column(elec_baseline_col,
                                             baseline_annualized)

                if has_valid_baseline_natural_gas_consumption:
                    baseline_annualized_kwh = (
                        baseline_annualized[0]*29.3001,
                        baseline_annualized[1]*29.3001,
                        baseline_annualized[2]*29.3001,
                        baseline_annualized[3]
                    )
                    update_derivative_column(fuel_baseline_col,
                                             baseline_annualized_kwh)
                    update_derivative_column(gas_baseline_col,
                                             baseline_annualized)

                if has_valid_reporting_electricity_consumption:
                    update_derivative_column(fuel_reporting_col,
                                             reporting_annualized)
                    update_derivative_column(elec_reporting_col,
                                             reporting_annualized)

                if has_valid_reporting_natural_gas_consumption:
                    reporting_annualized_kwh = (
                        reporting_annualized[0]*29.3001,
                        reporting_annualized[1]*29.3001,
                        reporting_annualized[2]*29.3001,
                        reporting_annualized[3]
                    )
                    update_derivative_column(fuel_reporting_col,
                                             reporting_annualized_kwh)
                    update_derivative_column(gas_reporting_col,
                                             reporting_annualized)

                if has_valid_baseline_generation:
                    update_derivative_column(solar_baseline_col,
                                             baseline_annualized)

                if has_valid_reporting_generation:
                    update_derivative_column(solar_reporting_col,
                                             reporting_annualized)

        return project_derivatives
