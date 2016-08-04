import logging

from eemeter.processors.interventions import get_modeling_period_set
from eemeter.processors.location import (
    get_weather_source,
    get_weather_normal_source,
)
from eemeter.processors.dispatchers import get_energy_modeling_dispatches
from eemeter.ee.derivatives import annualized_weather_normal, gross_predicted

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
        '''

        modeling_period_set = get_modeling_period_set(project.interventions)

        if weather_source is None:
            weather_source = get_weather_source(project)
        else:
            logger.info("Using supplied weather_source")

        if weather_normal_source is None:
            weather_normal_source = get_weather_normal_source(project)
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
                        weather_normal_source=weather_normal_source)
                    if awn is not None:
                        period_derivatives["BASELINE"].update(awn)

                    gp = modeled_energy_trace.compute_derivative(
                        baseline_label,
                        gross_predicted,
                        weather_source=weather_source,
                        reporting_period=reporting_period)
                    if gp is not None:
                        period_derivatives["BASELINE"].update(gp)

                if reporting_output["status"] == "SUCCESS":
                    awn = modeled_energy_trace.compute_derivative(
                        reporting_label,
                        annualized_weather_normal,
                        weather_normal_source=weather_normal_source)
                    if awn is not None:
                        period_derivatives["REPORTING"].update(awn)

                    gp = modeled_energy_trace.compute_derivative(
                        reporting_label,
                        gross_predicted,
                        weather_source=weather_source,
                        reporting_period=reporting_period)
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

                        if reporting_output is None:
                            reporting_output = (0.0, 0.0, 0.0, 0)

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
