from eemeter.processors.dispatchers import EnergyModelerDispatcher
from eemeter.processors.location import WeatherSourceMatcher
from eemeter.processors.interventions import EEInterventionModelingPeriodProcessor


class EnergyEfficiencyMeter(object):

    def __init__(self, settings=None):
        if settings is None:
            self.settings = {}
        self.settings = settings

    def get_modeling_period_processor(self):
        return EEInterventionModelingPeriodProcessor()

    def get_weather_source_matcher(self):
        return WeatherSourceMatcher()

    def get_energy_modeler_dispatcher(self):
        return EnergyModelerDispatcher()

    def get_energy_modelers(self, dispatches, weather_source):

        energy_modelers = {}
        for (energy_modeler, modeling_period_label, trace_label), validation_errors in dispatches:

            energy_modelers[(modeling_period_label, trace_label)] = energy_modeler

            # do something with validation errors
            if energy_modeler is None:
                print(validation_errors)
                continue

            # get basic input dataframe.
            input_df = energy_modeler.create_model_input(weather_source)

            # fit, params will be saved.
            energy_modeler.evaluate(input_df)

            if energy_modeler.invalid_fit:
                # this is not the right way to handle this
                energy_modelers[(modeling_period_label, trace_label)] = None

        return energy_modelers

    def get_energy_model_derivatives(self, energy_modelers, weather_normal_source):
        derivatives = {}
        for selector, energy_modeler in energy_modelers.items():
            derivatives[selector] = {}
            output = annualized_weather_normal(energy_modeler, weather_normal_source)
            derivatives[selector].update(output)
            if energy_modeler is None:
                derivatives[selector].update({
                    "r2": None,
                    "rmse": None,
                    "cvrmse": None,
                    "upper": None,
                    "lower": None,
                    "n": None,
                    "model_params": None,
                })
            else:
                derivatives[selector].update({
                    "r2": energy_modeler.model.r2,
                    "rmse": energy_modeler.model.rmse,
                    "cvrmse": energy_modeler.model.cvrmse,
                    "upper": energy_modeler.model.upper,
                    "lower": energy_modeler.model.lower,
                    "n": energy_modeler.model.n,
                    "model_params": energy_modeler.model.params,
                })
        return derivatives

    def get_project_derivatives(self, modeling_period_set, trace_set,
                                energy_modelers, energy_model_derivatives):

        project_derivatives = defaultdict(lambda: (0, (0,0), 0))

        for (baseline_label, baseline_period), (reporting_label, reporting_period) \
                in modeling_period_set.get_modeling_period_groups():
            for trace_label, trace in project.trace_set.get_traces():

                # baseline model
                def _get_baseline(label):
                    baseline_model_derivatives = energy_model_derivatives.get((baseline_label, trace_label), None)
                    if baseline_model_derivatives is not None:
                        return baseline_model_derivatives.get(label, None)

                # reporting model
                def _get_reporting(label):
                    reporting_model_derivatives = energy_model_derivatives.get((reporting_label, trace_label), None)
                    if reporting_model_derivatives is not None:
                        return reporting_model_derivatives.get(label, None)

                def _add_errors(errors1, errors2):
                    mean1, (lower1, upper1), n1 = errors1
                    mean2, (lower2, upper2), n2 = errors2

                    mean = mean1 + mean2
                    lower = (lower1**2 + lower2**2)**0.5
                    upper = (upper1**2 + upper2**2)**0.5
                    n = n1 + n2
                    return (mean, (lower, upper), n)

                baseline_annualized = _get_baseline("annualized_weather_normal")
                reporting_annualized = _get_reporting("annualized_weather_normal")

                has_valid_baseline_consumption = (
                    baseline_annualized is not None
                    and reporting_annualized is not None
                    and trace.interpretation == "CONSUMPTION_SUPPLIED"
                )
                has_valid_reporting_consumption = (
                    baseline_annualized is not None
                    and reporting_annualized is not None
                    and trace.interpretation == "CONSUMPTION_SUPPLIED"
                )
                has_valid_baseline_generation = (
                    baseline_annualized is not None
                    and trace.interpretation == "ON_SITE_GENERATION_UNCONSUMED"
                )
                has_valid_reporting_generation = (
                    reporting_annualized is not None
                    and trace.interpretation == "ON_SITE_GENERATION_UNCONSUMED"
                )

                if has_valid_baseline_consumption:

                    if trace.fuel == "electricity" and trace.unit == "kWh":

                        project_derivatives["total_baseline_normal_annual_fuel_consumption_kWh"] = \
                                _add_errors(project_derivatives["total_baseline_normal_annual_fuel_consumption_kWh"], baseline_annualized)
                        project_derivatives["total_baseline_normal_annual_electricity_consumption_kWh"] = \
                            _add_errors(project_derivatives["total_baseline_normal_annual_electricity_consumption_kWh"], baseline_annualized)

                    elif trace.fuel == "natural_gas" and trace.unit == "therm":

                        project_derivatives["total_baseline_normal_annual_fuel_consumption_kWh"] = \
                                _add_errors(project_derivatives["total_baseline_normal_annual_fuel_consumption_kWh"],
                                            (baseline_annualized[0]*29.3001,
                                             (baseline_annualized[1][0]*29.3001, baseline_annualized[1][1]*29.3001),
                                             baseline_annualized[2]))
                        project_derivatives["total_baseline_normal_annual_natural_gas_consumption_therms"] = \
                                _add_errors(project_derivatives["total_baseline_normal_annual_natural_gas_consumption_therms"], baseline_annualized)

                if has_valid_reporting_consumption:
                    if trace.fuel == "electricity" and trace.unit == "kWh":

                        project_derivatives["total_reporting_normal_annual_fuel_consumption_kWh"] = \
                            _add_errors(project_derivatives["total_reporting_normal_annual_fuel_consumption_kWh"], reporting_annualized)
                        project_derivatives["total_reporting_normal_annual_electricity_consumption_kWh"] = \
                            _add_errors(project_derivatives["total_reporting_normal_annual_electricity_consumption_kWh"], reporting_annualized)

                    elif trace.fuel == "natural_gas" and trace.unit == "therm":

                        project_derivatives["total_reporting_normal_annual_fuel_consumption_kWh"] = \
                            _add_errors(project_derivatives["total_reporting_normal_annual_fuel_consumption_kWh"],
                                            (reporting_annualized[0]*29.3001,
                                             (reporting_annualized[1][0]*29.3001, reporting_annualized[1][1]*29.3001),
                                             reporting_annualized[2]))
                        project_derivatives["total_reporting_normal_annual_natural_gas_consumption_therms"] = \
                            _add_errors(project_derivatives["total_reporting_normal_annual_natural_gas_consumption_therms"], reporting_annualized)

                if has_valid_baseline_generation:

                    project_derivatives["total_baseline_normal_annual_solar_generation_kWh"] = \
                        _add_errors(project_derivatives["total_baseline_normal_annual_solar_generation_kWh"], baseline_annualized)

                if has_valid_reporting_generation:

                    project_derivatives["total_reporting_normal_annual_solar_generation_kWh"] = \
                        _add_errors(project_derivatives["total_reporting_normal_annual_solar_generation_kWh"], reporting_annualized)

        return project_derivatives

    def evaluate(self, project):
        modeling_period_set, validation_errors = (
            self.get_modeling_period_processor()
            .get_modeling_period_set(project.interventions)
        )

        weather_source, validation_errors = (
            self.get_climate_data_matcher()
            .get_weather_source(project.location)
        )

        weather_normal_source, validation_errors = (
            self.get_climate_data_matcher()
            .get_weather_normal_source(project.location)
        )

        # do something with validation errors

        dispatches = (
            self.get_energy_modeler_dispatcher()
            .dispatch_energy_modelers(modeling_period_set, project.trace_set)
        )

        # pre checks?

        # iterate over energy modelers

        energy_modelers = self.get_energy_modelers(dispatches, weather_source)

        energy_model_derivatives = self.get_energy_model_derivatives(
            energy_modelers, weather_normal_source)

        # decide which internal model to use
        # validate inputs

        # energy_modeler.predict(df)

        # compute pre-modeling trace outputs and validations

        # if able to, run each energy modeler step and save model period and trace labels.

        # compute post-modeling trace outputs and validations

        project_derivatives = self.get_project_derivatives(
            modeling_period_set, project.trace_set, energy_modelers, energy_model_derivatives)

        output_data = {
            "project": {
                "modeled_trace_selectors": list(energy_modelers.keys()),
                "trace_descriptors": {
                    trace_id: (trace.fuel, trace.interpretation)
                    for trace_id, trace in project.trace_set.get_traces()
                },
                "modeling_periods": [
                    name for name, period in modeling_period_set.get_modeling_periods()
                ],
                "modeling_period_groups": [
                    (baseline_name, reporting_name)
                    for (baseline_name, _), (reporting_name, _) in modeling_period_set.get_modeling_period_groups()
                ]
            },
            "modeled_traces": energy_model_derivatives,
        }
        for key, value in project_derivatives.items():
            output_data["project"].update({key: value})
        return output_data
