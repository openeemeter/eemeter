from eemeter.meter.base import YamlDefinedMeter
from datetime import datetime
import pytz

bpi_meter_yaml = """
!obj:eemeter.meter.Sequence {
    sequence: [
        !obj:eemeter.meter.EstimatedReadingConsolidationMeter {
            input_mapping: { consumption_data: {} },
            output_mapping: { consumption_data_no_estimated: {} },
        },
        !obj:eemeter.meter.NormalAnnualHDD {
            base: !setting hdd_base,
            temperature_unit_str: !setting temperature_unit_str,
            input_mapping: { weather_normal_source: {} },
            output_mapping: { normal_annual_hdd: { name: hdd_tmy } },
        },
        !obj:eemeter.meter.NormalAnnualCDD {
            base: !setting cdd_base,
            temperature_unit_str: !setting temperature_unit_str,
            input_mapping: { weather_normal_source: {} },
            output_mapping: { normal_annual_cdd: { name: cdd_tmy } },
        },
        !obj:eemeter.meter.RecentReadingMeter {
            input_mapping: {
                consumption_data: { name: consumption_data_no_estimated }
            },
            output_mapping: { n_days: { name: n_days_since_reading } }
        },
        !obj:eemeter.meter.TimeSpanMeter {
            input_mapping: { consumption_data: { name: consumption_data_no_estimated } },
            output_mapping: { time_span: {} }
        },
        !obj:eemeter.meter.TotalHDDMeter {
            base: !setting hdd_base,
            temperature_unit_str: !setting temperature_unit_str,
            input_mapping: {
                consumption_data: { name: consumption_data_no_estimated },
                weather_source: {},
            },
            output_mapping: { total_hdd: {} }
        },
        !obj:eemeter.meter.TotalCDDMeter {
            base: !setting cdd_base,
            temperature_unit_str: !setting temperature_unit_str,
            input_mapping: {
                consumption_data: { name: consumption_data_no_estimated },
                weather_source: {},
            },
            output_mapping: { total_cdd: {} }
        },
        !obj:eemeter.meter.NPeriodsMeetingHDDPerDayThreshold {
            base: !setting hdd_base,
            temperature_unit_str: !setting temperature_unit_str,
            operation: ">",
            proportion: 0.0032876712,
            input_mapping: {
                consumption_data: { name: consumption_data_no_estimated },
                weather_source: {},
                hdd: { name: hdd_tmy, },
            },
            output_mapping: { n_periods: { name: n_periods_high_hdd_per_day }, }
        },
        !obj:eemeter.meter.NPeriodsMeetingHDDPerDayThreshold {
            base: !setting hdd_base,
            temperature_unit_str: !setting temperature_unit_str,
            operation: "<",
            proportion: .00054794521,
            input_mapping: {
                consumption_data: { name: consumption_data_no_estimated },
                weather_source: {},
                hdd: { name: hdd_tmy, },
            },
            output_mapping: { n_periods: { name: n_periods_low_hdd_per_day }, }
        },
        !obj:eemeter.meter.NPeriodsMeetingCDDPerDayThreshold {
            base: !setting cdd_base,
            temperature_unit_str: !setting temperature_unit_str,
            operation: ">",
            proportion: 0.0032876712,
            input_mapping: {
                consumption_data: { name: consumption_data_no_estimated },
                weather_source: {},
                cdd: { name: cdd_tmy, },
            },
            output_mapping: { n_periods: { name: n_periods_high_cdd_per_day }, }
        },
        !obj:eemeter.meter.NPeriodsMeetingCDDPerDayThreshold {
            base: !setting cdd_base,
            temperature_unit_str: !setting temperature_unit_str,
            operation: "<",
            proportion: .00054794521,
            input_mapping: {
                consumption_data: { name: consumption_data_no_estimated },
                weather_source: {},
                cdd: { name: cdd_tmy, },
            },
            output_mapping: { n_periods: { name: n_periods_low_cdd_per_day}, }
        },
        !obj:eemeter.meter.ConsumptionDataAttributes {
            input_mapping: { consumption_data: { name: consumption_data_no_estimated, }, },
            output_mapping: {
                fuel_type: {},
                unit_name: { name: energy_unit_str }
            }
        },
        !obj:eemeter.meter.Switch {
            target: { name: fuel_type },
            cases: {
                electricity: !obj:eemeter.meter.TemperatureSensitivityParameterOptimizationMeter {
                    temperature_unit_str: !setting temperature_unit_str,
                    model: !obj:eemeter.models.AverageDailyTemperatureSensitivityModel {
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
                        consumption_data: { name: consumption_data_no_estimated, },
                        weather_source: {},
                        energy_unit_str: {},
                    },
                    output_mapping: {
                        average_daily_usages: { name: average_daily_usages_bpi2400 },
                        estimated_average_daily_usages: { name: estimated_average_daily_usages_bpi2400 },
                        temp_sensitivity_params: { name: temp_sensitivity_params_bpi2400 },
                    },
                },
                natural_gas: !obj:eemeter.meter.TemperatureSensitivityParameterOptimizationMeter {
                    temperature_unit_str: !setting temperature_unit_str,
                    model: !obj:eemeter.models.AverageDailyTemperatureSensitivityModel {
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
                        consumption_data: { name: consumption_data_no_estimated, },
                        weather_source: {},
                        energy_unit_str: {},
                    },
                    output_mapping: {
                        average_daily_usages: { name: average_daily_usages_bpi2400 },
                        estimated_average_daily_usages: { name: estimated_average_daily_usages_bpi2400 },
                        temp_sensitivity_params: { name: temp_sensitivity_params_bpi2400 },
                    },
                },
            },
        },
        !obj:eemeter.meter.CVRMSE {
            input_mapping: {
                y: { name: average_daily_usages_bpi2400 },
                y_hat: { name: estimated_average_daily_usages_bpi2400 },
                params: { name: temp_sensitivity_params_bpi2400 },
            },
            output_mapping: { cvrmse: {} },
        },
        !obj:eemeter.meter.Switch {
            target: { name: fuel_type },
            cases: {
                electricity: !obj:eemeter.meter.MeetsThresholds {
                    equations: [
                        [time_span, ">=", 1, 330, 0, spans_330_days],
                        [time_span, ">", 1, 184, 0, spans_184_days],
                        [total_hdd, ">", .5, hdd_tmy, 0, has_enough_total_hdd],
                        [total_cdd, ">", .5, cdd_tmy, 0, has_enough_total_cdd],
                        [n_days_since_reading, "<", 1, 360, 0, has_recent_reading],
                        [n_periods_high_hdd_per_day, ">=", 1, 1, 0, has_enough_periods_with_high_hdd_per_day],
                        [n_periods_low_hdd_per_day, ">=", 1, 1, 0, has_enough_periods_with_low_hdd_per_day],
                        [n_periods_high_cdd_per_day, ">=", 1, 1, 0, has_enough_periods_with_high_cdd_per_day],
                        [n_periods_low_cdd_per_day, ">=", 1, 1, 0, has_enough_periods_with_low_cdd_per_day],
                        [cvrmse, "<=", 1, 20, 0, meets_cvrmse_limit],
                    ],
                    input_mapping: {
                        time_span: {},
                        total_hdd: {},
                        hdd_tmy: {},
                        total_cdd: {},
                        cdd_tmy: {},
                        n_days_since_reading: {},
                        n_periods_high_hdd_per_day: {},
                        n_periods_low_hdd_per_day: {},
                        n_periods_high_cdd_per_day: {},
                        n_periods_low_cdd_per_day: {},
                        cvrmse: {},
                    },
                    output_mapping: {
                        spans_330_days: {},
                        spans_184_days: {},
                        has_enough_total_hdd: {},
                        has_enough_total_cdd: {},
                        has_recent_reading: {},
                        has_enough_periods_with_high_hdd_per_day: {},
                        has_enough_periods_with_low_hdd_per_day: {},
                        has_enough_periods_with_high_cdd_per_day: {},
                        has_enough_periods_with_low_cdd_per_day: {},
                        meets_cvrmse_limit: {},
                    },
                },
                natural_gas: !obj:eemeter.meter.MeetsThresholds {
                    equations: [
                        [time_span, ">=", 1, 330, 0, spans_330_days],
                        [time_span, ">", 1, 184, 0, spans_184_days],
                        [total_hdd, ">", .5, hdd_tmy, 0, has_enough_total_hdd],
                        [n_days_since_reading, "<", 1, 360, 0, has_recent_reading],
                        [n_periods_high_hdd_per_day, ">=", 1, 1, 0, has_enough_periods_with_high_hdd_per_day],
                        [n_periods_low_hdd_per_day, ">=", 1, 1, 0, has_enough_periods_with_low_hdd_per_day],
                        [cvrmse, "<=", 1, 20, 0, meets_cvrmse_limit],
                    ],
                    input_mapping: {
                        time_span: {},
                        total_hdd: {},
                        hdd_tmy: {},
                        n_days_since_reading: {},
                        n_periods_high_hdd_per_day: {},
                        n_periods_low_hdd_per_day: {},
                        cvrmse: {},
                    },
                    auxiliary_outputs: {
                        has_enough_total_cdd: true,
                        has_enough_periods_with_high_cdd_per_day: true,
                        has_enough_periods_with_low_cdd_per_day: true,
                    },
                    output_mapping: {
                        spans_330_days: {},
                        spans_184_days: {},
                        has_enough_total_hdd: {},
                        has_enough_total_cdd: {},
                        has_recent_reading: {},
                        has_enough_periods_with_high_hdd_per_day: {},
                        has_enough_periods_with_low_hdd_per_day: {},
                        has_enough_periods_with_high_cdd_per_day: {},
                        has_enough_periods_with_low_cdd_per_day: {},
                        meets_cvrmse_limit: {},
                    },
                },
            }
        },
        !obj:eemeter.meter.And {
            inputs: [
                has_enough_total_hdd,
                has_enough_periods_with_high_hdd_per_day,
                has_enough_periods_with_low_hdd_per_day,
            ],
            input_mapping: {
                has_enough_total_hdd: {},
                has_enough_periods_with_high_hdd_per_day: {},
                has_enough_periods_with_low_hdd_per_day: {},
            },
            output_mapping: { output: { name: has_enough_hdd, }, },
        },
        !obj:eemeter.meter.And {
            inputs: [
                has_enough_total_cdd,
                has_enough_periods_with_high_cdd_per_day,
                has_enough_periods_with_low_cdd_per_day,
            ],
            input_mapping: {
                has_enough_total_cdd: {},
                has_enough_periods_with_high_cdd_per_day: {},
                has_enough_periods_with_low_cdd_per_day: {},
            },
            output_mapping: { output: { name: has_enough_cdd, }, },
        },
        !obj:eemeter.meter.And {
            inputs: [
                has_enough_hdd,
                has_enough_cdd
            ],
            input_mapping: {
                has_enough_hdd: {},
                has_enough_cdd: {},
            },
            output_mapping: { output: { name: has_enough_hdd_cdd }, }
        },
        !obj:eemeter.meter.And {
            inputs: [
                spans_184_days,
                has_enough_hdd_cdd
            ],
            input_mapping: {
                spans_184_days: {},
                has_enough_hdd_cdd: {},
            },
            output_mapping: { output: { name: spans_183_days_and_has_enough_hdd_cdd, }, }
        },
        !obj:eemeter.meter.Or {
            inputs: [
                spans_330_days,
                spans_183_days_and_has_enough_hdd_cdd
            ],
            input_mapping: {
                spans_330_days: {},
                spans_183_days_and_has_enough_hdd_cdd: {},
            },
            output_mapping: { output: { name: has_enough_data } }
        },
        !obj:eemeter.meter.And {
            inputs: [
                has_recent_reading,
                has_enough_data,
                meets_cvrmse_limit,
            ],
            input_mapping: {
                has_recent_reading: {},
                has_enough_data: {},
                meets_cvrmse_limit: {},
            },
            output_mapping: { output: { name: meets_model_calibration_utility_bill_criteria }, }
        }
    ]
}
"""

class BPI_2400_S_2012_ModelCalibrationUtilityBillCriteria(YamlDefinedMeter):
    """Implementation of BPI-2400-S-2012 section 3.2.2.

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

        - hdd_base (float):
          Base for Heating Degree Day calculations.
          defaults to 65 degF
        - cdd_base (float):
          Base for Cooling Degree Day calculations.
          defaults to 65 degF
    """

    def __init__(self, temperature_unit_str, **kwargs):

        if temperature_unit_str not in ["degF","degC"]:
            raise ValueError("Invalid temperature_unit_str: should be one of 'degF' or 'degC'.")

        self.temperature_unit_str = temperature_unit_str

        super(BPI_2400_S_2012_ModelCalibrationUtilityBillCriteria, self).__init__(**kwargs)

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
        return bpi_meter_yaml

    def evaluate(self, data_collection):
        """Evaluates utility bills for compliance with criteria specified in
        ANSI/BPI-2400-S-2012 section 3.2.2.

        Parameters
        ----------
        consumption_history : eemeter.consumption.ConsumptionHistory
            All available billing data (of all fuel types) available for the
            target project. Estimated bills must be flagged.
        weather_source : eemeter.weather.WeatherSourceBase
            Weather data should come from a source as geographically and
            climatically similar to the target project as possible.
        weather_normal_source : eemeter.weather.WeatherSourceBase with eemeter.weather.WeatherNormalMixin
            Weather normal data should come from a source as geographically and
            climatically similar to the target project as possible.
        since_date : datetime.datetime, optional
            The date from which to count days since most recent reading;
            defaults to datetime.now(pytz.utc).

        Returns
        -------
        out : eemeter.meter.DataCollection

            - *"average_daily_usages_bpi2400"* : Average usage per
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
            - *"model_params"* : Fitted temperature
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

        """
        return super(BPI_2400_S_2012_ModelCalibrationUtilityBillCriteria, self).evaluate(data_collection)
