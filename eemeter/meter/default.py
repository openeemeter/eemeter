from eemeter.meter.base import YamlDefinedMeter

default_residential_meter_yaml = """
!obj:eemeter.meter.Sequence {
    sequence: [
        !obj:eemeter.meter.ProjectAttributes {
            input_mapping: { project: {} },
            output_mapping: {
                weather_source: {},
                weather_normal_source: {},
            },
        },
        !obj:eemeter.meter.ProjectConsumptionDataBaselineReporting {
            input_mapping: { project: {} },
            output_mapping: { consumption: {}, },
        },
        !obj:eemeter.meter.For {
            variable: { name: consumption_data_raw },
            iterable: { name: consumption },
            meter: !obj:eemeter.meter.Sequence {
                sequence: [
                    !obj:eemeter.meter.DownsampleConsumption {
                        freq: 'D',
                        input_mapping: { consumption_data: {name: consumption_data_raw} },
                        output_mapping: { consumption_downsampled: {name: consumption_data}, },
                    },
                    !obj:eemeter.meter.BPI_2400_S_2012_ModelCalibrationUtilityBillCriteria {
                        temperature_unit_str: !setting temperature_unit_str,
                        settings: {
                            temperature_unit_str: !setting temperature_unit_str,
                            electricity_baseload_low: !setting electricity_baseload_low,
                            electricity_baseload_x0: !setting electricity_baseload_x0,
                            electricity_baseload_high: !setting electricity_baseload_high,
                            electricity_heating_slope_low: !setting electricity_heating_slope_low,
                            electricity_heating_slope_x0: !setting electricity_heating_slope_x0,
                            electricity_heating_slope_high: !setting electricity_heating_slope_high,
                            electricity_cooling_slope_low: !setting electricity_cooling_slope_low,
                            electricity_cooling_slope_x0: !setting electricity_cooling_slope_x0,
                            electricity_cooling_slope_high: !setting electricity_cooling_slope_high,
                            natural_gas_baseload_low: !setting natural_gas_baseload_low,
                            natural_gas_baseload_x0: !setting natural_gas_baseload_x0,
                            natural_gas_baseload_high: !setting natural_gas_baseload_high,
                            natural_gas_heating_slope_low: !setting natural_gas_heating_slope_low,
                            natural_gas_heating_slope_x0: !setting natural_gas_heating_slope_x0,
                            natural_gas_heating_slope_high: !setting natural_gas_heating_slope_high,
                            heating_balance_temp_low: !setting heating_balance_temp_low,
                            heating_balance_temp_x0: !setting heating_balance_temp_x0,
                            heating_balance_temp_high: !setting heating_balance_temp_high,
                            cooling_balance_temp_low: !setting cooling_balance_temp_low,
                            cooling_balance_temp_x0: !setting cooling_balance_temp_x0,
                            cooling_balance_temp_high: !setting cooling_balance_temp_high,
                            hdd_base: !setting hdd_base,
                            cdd_base: !setting cdd_base,
                        },
                        tagspace: ["bpi2400"],
                    },
                    !obj:eemeter.meter.Sequence {
                        sequence: [
                            !obj:eemeter.meter.Switch {
                                target: {
                                    name: fuel_type,
                                    tags: ["bpi2400"]
                                },
                                cases: {
                                    electricity: !obj:eemeter.meter.Sequence {
                                        sequence: [
                                            !obj:eemeter.meter.TemperatureSensitivityParameterOptimizationMeter {
                                                temperature_unit_str: !setting temperature_unit_str,
                                                model: !obj:eemeter.models.AverageDailyTemperatureSensitivityModel &electricity_model {
                                                    cooling: True,
                                                    heating: True,
                                                    initial_params: {
                                                        base_daily_consumption: !setting electricity_baseload_x0,
                                                        heating_slope: !setting electricity_heating_slope_x0,
                                                        cooling_slope: !setting electricity_cooling_slope_x0,
                                                        heating_balance_temperature: !setting heating_balance_temp_x0,
                                                        cooling_balance_temperature: !setting cooling_balance_temp_x0,
                                                    },
                                                    param_bounds: {
                                                        base_daily_consumption: [!setting electricity_baseload_low, !setting electricity_baseload_high],
                                                        heating_slope: [!setting electricity_heating_slope_low, !setting electricity_heating_slope_high],
                                                        cooling_slope: [!setting electricity_heating_slope_low, !setting electricity_cooling_slope_high],
                                                        heating_balance_temperature: [!setting heating_balance_temp_low, !setting heating_balance_temp_high],
                                                        cooling_balance_temperature: [!setting cooling_balance_temp_low, !setting cooling_balance_temp_high],
                                                    },
                                                },
                                                input_mapping: {
                                                    consumption_data: {},
                                                    weather_source: {},
                                                    energy_unit_str: {},
                                                },
                                                output_mapping: {
                                                    temp_sensitivity_params: { name: model_params },
                                                    average_daily_usages: {},
                                                    estimated_average_daily_usages: {},
                                                },
                                            },
                                            !obj:eemeter.meter.AnnualizedUsageMeter {
                                                temperature_unit_str: !setting temperature_unit_str,
                                                model: *electricity_model,
                                                input_mapping: {
                                                    model_params: {},
                                                    weather_normal_source: {},
                                                },
                                                output_mapping: {
                                                    annualized_usage: {},
                                                },
                                            },
                                        ]
                                    },
                                    natural_gas: !obj:eemeter.meter.Sequence {
                                        sequence: [
                                            !obj:eemeter.meter.TemperatureSensitivityParameterOptimizationMeter {
                                                temperature_unit_str: !setting temperature_unit_str,
                                                model: !obj:eemeter.models.AverageDailyTemperatureSensitivityModel &natural_gas_model {
                                                    cooling: False,
                                                    heating: True,
                                                    initial_params: {
                                                        base_daily_consumption: !setting natural_gas_baseload_x0,
                                                        heating_slope: !setting natural_gas_heating_slope_x0,
                                                        heating_balance_temperature: !setting heating_balance_temp_x0,
                                                    },
                                                    param_bounds: {
                                                        base_daily_consumption: [!setting natural_gas_baseload_low, !setting natural_gas_baseload_high],
                                                        heating_slope: [!setting natural_gas_heating_slope_low, !setting natural_gas_heating_slope_high],
                                                        heating_balance_temperature: [!setting heating_balance_temp_low, !setting heating_balance_temp_high],
                                                    },
                                                },
                                                input_mapping: {
                                                    consumption_data: {},
                                                    weather_source: {},
                                                    energy_unit_str: {},
                                                },
                                                output_mapping: {
                                                    temp_sensitivity_params: { name: model_params },
                                                    average_daily_usages: {},
                                                    estimated_average_daily_usages: {},
                                                },
                                            },
                                            !obj:eemeter.meter.AnnualizedUsageMeter {
                                                temperature_unit_str: !setting temperature_unit_str,
                                                model: *natural_gas_model,
                                                input_mapping: {
                                                    model_params: {},
                                                    weather_normal_source: {},
                                                },
                                                output_mapping: {
                                                    annualized_usage: {},
                                                },
                                            },
                                        ]
                                    },
                                },
                            },
                            !obj:eemeter.meter.RMSE {
                                input_mapping: {
                                    y: { name: average_daily_usages },
                                    y_hat: { name: estimated_average_daily_usages },
                                },
                                output_mapping: {
                                    rmse: {},
                                }
                            },
                            !obj:eemeter.meter.RSquared {
                                input_mapping: {
                                    y: { name: average_daily_usages },
                                    y_hat: { name: estimated_average_daily_usages },
                                },
                                output_mapping: {
                                    r_squared: {},
                                }
                            },
                        ],
                    },
                ]
            },
        },
        !obj:eemeter.meter.ProjectFuelTypes {
            input_mapping: { project: {} },
            output_mapping: { fuel_types: {} },
        },
        !obj:eemeter.meter.For {
            variable: { name: active_fuel_type },
            iterable: { name: fuel_types },
            meter: !obj:eemeter.meter.FuelTypeTagFilter {
                fuel_type_search_name: active_fuel_type,
                input_mapping: {
                    weather_source: {},
                    active_fuel_type: {},
                },
                meter: !obj:eemeter.meter.Switch {
                    target: {
                        name: active_fuel_type,
                        tags: [],
                    },
                    cases: {
                        electricity: !obj:eemeter.meter.GrossSavingsMeter {
                            temperature_unit_str: !setting temperature_unit_str,
                            model: *electricity_model,
                            input_mapping: {
                                model_params_baseline: {
                                    name: model_params,
                                    tags: [ baseline ]
                                },
                                consumption_data_reporting : {
                                    name: consumption_data_no_estimated,
                                    tags: [ reporting ]
                                },
                                weather_source: {},
                                energy_unit_str : { tags: [ baseline ] },
                            },
                            output_mapping: {
                                gross_savings: {},
                            },
                        },
                        natural_gas: !obj:eemeter.meter.GrossSavingsMeter {
                            temperature_unit_str: !setting temperature_unit_str,
                            model: *natural_gas_model,
                            input_mapping: {
                                model_params_baseline: {
                                    name: model_params,
                                    tags: [ baseline ]
                                },
                                consumption_data_reporting : {
                                    name: consumption_data_no_estimated,
                                    tags: [ reporting ]
                                },
                                weather_source: {},
                                energy_unit_str : { tags: [ baseline ] },
                            },
                            output_mapping: {
                                gross_savings: {},
                            },
                        },
                    },
                },
            },
        },
    ]
}"""

class DefaultResidentialMeter(YamlDefinedMeter):
    """Implementation of the core EE-Meter savings evaluation method with
    defualt settings for evaluation of residential energy efficiency projects.

    Parameters
    ----------
    temperature_unit_str : {"degF", "degC"}, default "degC"
        Temperature unit to use throughout meter in degree-day calculations and
        parameter optimizations.
    settings : dict
        - electricity_baseload_low (float):
          Lowest baseload in parameter optimization for electricity;
          defaults to 0 energy units/day
        - electricity_baseload_x0 (float):
          Initial baseload in parameter optimization for electricity;
          defaults to 0 energy units/day
        - electricity_baseload_high (float):
          Highest baseload in parameter optimization for electricity;
          defaults to 1000 energy units/day


        - electricity_heating_slope_low (float):
          Lowest heating slope in parameter optimization for electricity;
          defaults to 0 energy units/degF/day
        - electricity_heating_slope_x0 (float):
          Initial heating slope in parameter optimization for electricity;
          defaults to 0 energy units/degF/day
        - electricity_heating_slope_high (float):
          Highest heating slope in parameter optimization for electricity;
          defaults to or 100 energy units/degF/day

        - electricity_cooling_slope_low (float):
          Lowest cooling slope in parameter optimization for electricity;
          defaults to 0 energy units/degF/day
        - electricity_cooling_slope_x0 (float):
          Initial cooling slope in parameter optimization for electricity;
          defaults to 0 energy units/degF/day
        - electricity_cooling_slope_high (float):
          Highest cooling slope in parameter optimization for electricity;
          defaults to 100 energy units/degF/day


        - natural_gas_baseload_low (float):
          Lowest baseload in parameter optimization for natural gas;
          defaults to 0 energy units/day
        - natural_gas_baseload_x0 (float):
          Initial baseload in parameter optimization for natural gas;
          defaults to 0 energy units/day
        - natural_gas_baseload_high (float):
          Highest baseload in parameter optimization for natural gas;
          defaults to 1000 energy units/day

        - natural_gas_heating_slope_low (float):
          Lowest heating slope in parameter optimization for natural gas;
          defaults to 0 energy units/degF/day
        - natural_gas_heating_slope_x0 (float):
          Initial heating slope in parameter optimization for natural gas;
          defaults to 0 energy units/degF/day
        - natural_gas_heating_slope_high (float):
          Highest heating slope in parameter optimization for natural gas;
          defaults to 100 energy units/degF/day

        - heating_balance_temp_low (float):
          Lowest heating balance temperature in parameter optimization;
          defaults to 55 degF
        - heating_balance_temp_x0 (float):
          Initial heating balance temperature in parameter optimization;
          defaults to 60 degF
        - heating_balance_temp_high (float):
          Highest heating balance temperature in parameter optimization;
          defaults to 70 degF
        - cooling_balance_temp_low (float):
          Lowest cooling balance temperature in parameter optimization;
          defaults to 60 degF
        - cooling_balance_temp_x0 (float):
          Initial cooling balance temperature in parameter optimization;
          defaults to 70 degF
        - cooling_balance_temp_high (float):
          Highest cooling balance temperature in parameter optimization;
          defaults to 75 degF
    """

    def __init__(self, temperature_unit_str="degC", **kwargs):

        if temperature_unit_str not in ["degF","degC"]:
            raise ValueError("Invalid temperature_unit_str: should be one of 'degF' or 'degC'.")

        self.temperature_unit_str = temperature_unit_str

        super(DefaultResidentialMeter, self).__init__(**kwargs)

    def default_settings(self):

        def degF_to_degC(F):
            return (F - 32.) * 5. / 9.

        def convert_temp_degF_to_target(temp_degF):
            if self.temperature_unit_str == "degF":
                return temp_degF
            else:
                return degF_to_degC(temp_degF)

        def convert_slope_degF_to_target(slope_degF):
            if self.temperature_unit_str == "degF":
                return slope_degF
            else:
                return slope_degF * 1.8

        settings = {
                "temperature_unit_str": self.temperature_unit_str,
                "electricity_baseload_low": 0,
                "electricity_baseload_x0": 0,
                "electricity_baseload_high": 1000,
                "electricity_heating_slope_low": convert_slope_degF_to_target(0),
                "electricity_heating_slope_x0": convert_slope_degF_to_target(0),
                "electricity_heating_slope_high": convert_slope_degF_to_target(1000),
                "electricity_cooling_slope_low": convert_slope_degF_to_target(0),
                "electricity_cooling_slope_x0": convert_slope_degF_to_target(0),
                "electricity_cooling_slope_high": convert_slope_degF_to_target(1000),

                "natural_gas_baseload_low": 0,
                "natural_gas_baseload_x0": 0,
                "natural_gas_baseload_high": 1000,
                "natural_gas_heating_slope_low":  convert_slope_degF_to_target(0),
                "natural_gas_heating_slope_x0":  convert_slope_degF_to_target(0),
                "natural_gas_heating_slope_high":  convert_slope_degF_to_target(1000),

                "heating_balance_temp_low": convert_temp_degF_to_target(55),
                "heating_balance_temp_x0": convert_temp_degF_to_target(60),
                "heating_balance_temp_high": convert_temp_degF_to_target(70),
                "cooling_balance_temp_low": convert_temp_degF_to_target(60),
                "cooling_balance_temp_x0": convert_temp_degF_to_target(70),
                "cooling_balance_temp_high": convert_temp_degF_to_target(75),

                "hdd_base": convert_temp_degF_to_target(65),
                "cdd_base": convert_temp_degF_to_target(65),
        }
        return settings

    def validate_settings(self, settings):

        if not 0 <= settings["electricity_baseload_low"] <= \
                settings["electricity_baseload_x0"] <= settings["electricity_baseload_high"]:
            message = "Electricity baseload parameter limits must be such " \
                    "that 0 <= low <= x0 <= high, but found low={}, x0={}, " \
                    "high={}".format(settings["electricity_baseload_low"],
                            settings["electricity_baseload_x0"],
                            settings["electricity_baseload_high"])
            raise ValueError(message)
        if not 0 <= settings["electricity_heating_slope_low"] <= \
                settings["electricity_heating_slope_x0"] <= settings["electricity_heating_slope_high"]:
            message = "Electricity heating slope parameter limits must be such " \
                    "that 0 <= low <= x0 <= high, but found low={}, x0={}, " \
                    "high={}".format(settings["electricity_heating_slope_low"],
                            settings["electricity_heating_slope_x0"],
                            settings["electricity_heating_slope_high"])
            raise ValueError(message)
        if not 0 <= settings["electricity_cooling_slope_low"] <= \
                settings["electricity_cooling_slope_x0"] <= settings["electricity_cooling_slope_high"]:
            message = "Electricity cooling slope parameter limits must be such " \
                    "that 0 <= low <= x0 <= high, but found low={}, x0={}, " \
                    "high={}".format(settings["electricity_cooling_slope_low"],
                            settings["electricity_cooling_slope_x0"],
                            settings["electricity_cooling_slope_high"])
            raise ValueError(message)

        if not 0 <= settings["natural_gas_baseload_low"] <= \
                settings["natural_gas_baseload_x0"] <= settings["natural_gas_baseload_high"]:
            message = "Natural gas baseload parameter limits must be such " \
                    "that 0 <= low <= x0 <= high, but found low={}, x0={}, " \
                    "high={}".format(settings["natural_gas_baseload_low"],
                            settings["natural_gas_baseload_x0"],
                            settings["natural_gas_baseload_high"])
            raise ValueError(message)
        if not 0 <= settings["natural_gas_heating_slope_low"] <= \
                settings["natural_gas_heating_slope_x0"] <= settings["natural_gas_heating_slope_high"]:
            message = "Natural gas heating slope parameter limits must be such " \
                    "that 0 <= low <= x0 <= high, but found low={}, x0={}, " \
                    "high={}".format(settings["natural_gas_heating_slope_low"],
                            settings["natural_gas_heating_slope_x0"],
                            settings["natural_gas_heating_slope_high"])
            raise ValueError(message)


        if not settings["heating_balance_temp_low"] <= settings["heating_balance_temp_x0"] <= settings["heating_balance_temp_high"]:
            raise ValueError("Heating balance temperature parameter limits must be such that low <= x0 <= high")
            message = "Heating balance temperature parameter limits must be such " \
                    "that low <= x0 <= high, but found low={}, x0={}, " \
                    "high={}".format(settings["heating_balance_temp_low"],
                            settings["heating_balance_temp_x0"],
                            settings["heating_balance_temp_high"])
            raise ValueError(message)
        if not settings["cooling_balance_temp_low"] <= settings["cooling_balance_temp_x0"] <= settings["cooling_balance_temp_high"]:
            message = "Cooling balance temperature parameter limits must be such " \
                    "that low <= x0 <= high, but found low={}, x0={}, " \
                    "high={}".format(settings["cooling_balance_temp_low"],
                            settings["cooling_balance_temp_x0"],
                            settings["cooling_balance_temp_high"])
            raise ValueError(message)

    @property
    def yaml(self):
        return default_residential_meter_yaml

    def evaluate(self, data_collection):
        """PRISM-style evaluation of temperature sensitivity and
        weather-normalized annual consumption (NAC) at the single-project
        level.

        .. code-block:: python

            meter = DefaultResidentialMeter()
            results = meter.evaluate(DataCollection(project=project))


        Parameters
        ----------
        project : eemeter.project.Project
            Container for single-project consumption data, baseline/reporting
            period specifications (retrofit dates), and location data (for
            matching with weather sources.

        Returns
        -------
        out : dict
            The following results are always available:

            - *"average_daily_usages"* : Average usage per
              day (kWh/day) for the consumption periods.
            - *"cdd_tmy"* : Total cooling degree days (base 65 degF or 18.33 degC)
              in a typical meteorological year (TMY3).
            - *"consumption_history_no_estimated"* : The input consumption history
              with estimated periods consolidated or removed.
            - *"cvrmse"* : The Coefficient of Variation of
              Root-mean-squared Error on the outputs of the usage model.
            - *"estimated_average_daily_usages"* : Average usage per day for
              the consumption_periods as estimated by the fitted temperature
              sensitivity model.
            - *"has_enough_cdd"* : A boolean indicating whether or not
              the consumption data covers (a) enough total CDD, (b)
              enough periods with low CDD, and (c) enough periods with high
              CDD.
            - *"has_enough_data"* : A boolean indicating whether or not
              the consumption data covers a period of at least 330
              days or a period of at least 184 days with enough CDD and HDD
              variation, as indicated by the results "has_enough_cdd" and
              "has_enough_hdd".
            - *"has_enough_data"* : A boolean indicating whether or not
              the consumption data covers a period of at least 330
              days or a period of at least 184 days with enough CDD and HDD
              variation, as indicated by the result "has_enough_hdd_cdd".
            - *"has_enough_hdd_cdd"* : A boolean indicating whether or
              not the electricity consumption data covers a period with enough
              variation in hdd and cdd; equivalent to the boolean value
              ("has_enough_cdd" and "has_enough_hdd").
            - *"has_enough_hdd"* : A boolean indicating whether or not
              the consumption data covers (a) enough total HDD, (b)
              enough periods with low HDD, and (c) enough periods with high
              HDD.
            - *"has_enough_periods_with_high_cdd_per_day"* : A boolean
              indicating whether or not the consumption data has
              enough periods with at least 1.2x average normal CDD/day (TMY3,
              base 65 degF or 18.33 degC).
            - *"has_enough_periods_with_high_hdd_per_day"* : A boolean
              indicating whether or not the consumption data has
              enough periods with at least 1.2x average normal HDD/day (TMY3,
              base 65 degF or 18.33 degC).
            - *"has_enough_periods_with_low_cdd_per_day"* : A boolean
              indicating whether or not the consumption data has
              enough periods with less than 0.2x average normal CDD/day (TMY3,
              base 65 degF or 18.33 degC).
            - *"has_enough_periods_with_low_hdd_per_day"* : A boolean
              indicating whether or not the consumption data has
              enough periods with less than 0.2x average normal HDD/day (TMY3,
              base 65 degF or 18.33 degC).
            - *"has_enough_total_cdd"* : A boolean indicating whether
              or not the total CDD during the total time span of the
              data is at least 0.5x normal annual CDD/day (TMY3,
              base 65 degF or 18.33 degC).
            - *"has_enough_total_hdd"* : A boolean indicating whether
              or not the total HDD during the total time span of the
              data is at least 0.5x normal annual HDD/day (TMY3,
              base 65 degF or 18.33 degC).
            - *"has_recent_reading"* : A boolean indicating whether or
              not there is valid (not missing) consumption data within 365 days
              of the last date in the consumption data.
            - *"hdd_tmy"* : Total heating degree days (base 65 degF or 18.33 degC)
              in a typical meteorological year (TMY3).
            - *"meets_cvrmse_limit"* : A boolean indicating whether or
              not the Coefficient of Variation of the Root-mean-square Error
              (CVRMSE) of a regression of consumption data against
              local observed HDD/CDD, as determined using equation 3.2.2.G.i
              of the ANSI/BPI-2400-S-2012 specification is less than 20.
            - *"meets_model_calibration_utility_bill_criteria"* : A
              boolean indicating whether or not consumption data
              acceptance criteria, as outlined in section 3.2.2 of the
              ANSI/BPI-2400-S-2012 specification, have been met.
            - *"n_periods_high_cdd_per_day"* : The number of
              consumption data periods with observed CDD greater
              than 1.2x average normal CDD/day (TMY3, base 65 degF or 18.33
              degC).
            - *"n_periods_low_cdd_per_day"* : The number of
              consumption data periods with observed CDD less than
              0.2x average normal CDD/day (TMY3, base 65 degF or 18.33 degC).
            - *"n_periods_low_hdd_per_day"* : The number of
              consumption data periods with observed HDD less than
              0.2x average normal CDD/day (TMY3, base 65 degF or 18.33 degC).
            - *"n_periods_high_cdd_per_day"* : The number of natural
              consumption data periods with observed CDD greater than 1.2x
              average normal CDD/day (TMY3, base 65 degF or 18.33 degC).
            - *"n_periods_high_hdd_per_day"* : The number of natural
              consumption data periods with observed HDD greater than 1.2x
              average normal CDD/day (TMY3, base 65 degF or 18.33 degC).
            - *"spans_183_days_and_has_enough_hdd_cdd"* : A boolean
              indicating whether or not consumption data spans at
              least 184 days and is associated with sufficient breadth and
              variation in observed HDD and CDD.
            - *"spans_184_days"* : A boolean indicating whether or not
              consumption data spans at least 184 days.
            - *"spans_330_days"* : A boolean indicating whether or not
              consumption data spans at least 330 days.
            - *"model_params"* : (BPI2400) Fitted temperature
              sensitivity parameters for HDD/CDD use model in an
              array of values with the following order:

              For electricty: [base_daily_consumption (kWh/day),
              heating_balance_temperature (degF or degC), heating_slope (kWh/HDD),
              cooling_balance_temperature (degF or degC), cooling_slope (kWh/CDD)].

              For natural_gas: [base_daily_consumption (kWh/day),
              heating_balance_temperature (degF or degC), heating_slope (kWh/HDD)].

            - *"time_span"* : Number of days between earliest available
              data and latest available data.
            - *"total_cdd"* : The total cooling degree days (base 65
              degF or 18.33 degC) observed during the all consumption data
              periods.
            - *"total_hdd"* : The total heating degree days (base 65
              degF or 18.33 degC) observed during the all consumption data
              periods.

            The following results are only available for specific fuel types if
            :code:`meets_model_calibration_utility_bill_criteria`
            is :code:`True`:

            - *"annualized_usage"* : Usage in a typical meteorological year, as
              estimated by the fitted hdd/cdd use model.
            - *"average_daily_usages"* : Average usage per day for the
              consumption periods.
            - *"estimated_average_daily_usages"* : Average usage per day for
              the consumption periods as estimated by the fitted temperature
              sensitivity model.
            - *"n_days"* : The number of days in each consumption period;
              used as weights in model fitting.
            - *"rmse"* : Root-mean-square error of fitted hdd/cdd use model
              estimations for all consumption periods.
            - *"r_squared"* : Coefficient of Determination (r^2) of fitted
              HDD/CDD use model estimations for all consumption periods.
            - *"model_params"* : Fitted temperature
              sensitivity parameters for HDD/CDD use model in an
              array of values with the following order:

              For electricty: [base_daily_consumption (kWh/day),
              heating_balance_temperature (degF or degC), heating_slope (kWh/HDD),
              cooling_balance_temperature (degF or degC), cooling_slope (kWh/CDD)].

              For natural_gas: [base_daily_consumption (kWh/day),
              heating_balance_temperature (degF or degC), heating_slope (kWh/HDD)].

        """
        return super(DefaultResidentialMeter, self).evaluate(data_collection)
