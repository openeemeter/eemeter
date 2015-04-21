from eemeter.meter.base import MeterBase
from eemeter.config.yaml_parser import load

class PRISMMeter(MeterBase):
    """Implementation of Princeton Scorekeeping Method.

    Parameters
    ----------
    temperature_unit_str : {"degF", "degC"}, default "degC"
        Temperature unit to use throughout meter in degree-day calculations and
        parameter optimizations.
    heating_ref_temp_low : float, default None
        Lowest heating reference temperature in parameter optimization;
        defaults to 14.44 degC or 58 degF
    heating_ref_temp_x0 : float
        Initial heating reference temperature in parameter optimization;
        defaults to 15.55 degC or 60 degF
    heating_ref_temp_high : float
        Highest heating reference temperature in parameter optimization;
        defaults to 18.88 degC or 66 degF
    cooling_ref_temp_low : float
        Lowest cooling reference temperature in parameter optimization;
        defaults to 17.77 degC or 64 degF
    cooling_ref_temp_x0 : float
        Initial cooling reference temperature in parameter optimization;
        defaults to 21.11 degC or 70 degF
    cooling_ref_temp_high : float
        Highest cooling reference temperature in parameter optimization;
        defaults to 22.22 degC or 72 degF
    electricity_heating_slope_high : float
        Highest heating slope in parameter optimization for electricity;
        defaults to 9 energy/degC or 5 energy/degF
    natural_gas_heating_slope_high : float
        Highest heating slope in parameter optimization for natural gas;
        defaults to 9 energy/degC or 5 energy/degF
    electricity_cooling_slope_high : float
        Highest cooling slope in parameter optimization for electricity;
        defaults to 9 energy/degC or 5 energy/degF
    """

    def __init__(self,temperature_unit_str="degC",
                      heating_ref_temp_low=None,
                      heating_ref_temp_x0=None,
                      heating_ref_temp_high=None,
                      cooling_ref_temp_low=None,
                      cooling_ref_temp_x0=None,
                      cooling_ref_temp_high=None,
                      electricity_heating_slope_high=None,
                      natural_gas_heating_slope_high=None,
                      electricity_cooling_slope_high=None,
                      **kwargs):
        super(PRISMMeter, self).__init__(**kwargs)

        if temperature_unit_str not in ["degF","degC"]:
            raise ValueError("Invalid temperature_unit_str: should be one of 'degF' or 'degC'.")

        self.temperature_unit_str = temperature_unit_str
        self.heating_ref_temp_low=heating_ref_temp_low
        self.heating_ref_temp_x0=heating_ref_temp_x0
        self.heating_ref_temp_high=heating_ref_temp_high
        self.electricity_heating_slope_high=electricity_heating_slope_high
        self.natural_gas_heating_slope_high=natural_gas_heating_slope_high
        self.cooling_ref_temp_low=cooling_ref_temp_low
        self.cooling_ref_temp_x0=cooling_ref_temp_x0
        self.cooling_ref_temp_high=cooling_ref_temp_high
        self.electricity_cooling_slope_high=electricity_cooling_slope_high
        self.meter = load(self._meter_yaml())

    def _meter_yaml(self):

        def degF_to_degC(F):
            return (F - 32.) * 5. / 9.

        def convert_temp(temp_degF):
            if self.temperature_unit_str == "degF":
                return temp_degF
            else:
                return degF_to_degC(temp_degF)

        def convert_slope(slope_degF):
            if self.temperature_unit_str == "degF":
                return slope_degF
            else:
                return slope_degF * 1.8

        if self.heating_ref_temp_low is None:
            heating_ref_temp_low_degF = 58
            self.heating_ref_temp_low = convert_temp(heating_ref_temp_low_degF)

        if self.heating_ref_temp_x0 is None:
            heating_ref_temp_x0_degF = 60
            self.heating_ref_temp_x0 = convert_temp(heating_ref_temp_x0_degF)

        if self.heating_ref_temp_high is None:
            heating_ref_temp_high_degF = 66
            self.heating_ref_temp_high = convert_temp(heating_ref_temp_high_degF)

        if self.cooling_ref_temp_low is None:
            cooling_ref_temp_low_degF = 64
            self.cooling_ref_temp_low = convert_temp(cooling_ref_temp_low_degF)

        if self.cooling_ref_temp_x0 is None:
            cooling_ref_temp_x0_degF = 70
            self.cooling_ref_temp_x0 = convert_temp(cooling_ref_temp_x0_degF)

        if self.cooling_ref_temp_high is None:
            cooling_ref_temp_high_degF = 72
            self.cooling_ref_temp_high = convert_temp(cooling_ref_temp_high_degF)

        if self.electricity_heating_slope_high is None:
            electricity_heating_slope_high_degF = 5
            self.electricity_heating_slope_high = convert_slope(electricity_heating_slope_high_degF)

        if self.natural_gas_heating_slope_high is None:
            natural_gas_heating_slope_high_degF = 5
            self.natural_gas_heating_slope_high = convert_slope(natural_gas_heating_slope_high_degF)

        if self.electricity_cooling_slope_high is None:
            electricity_cooling_slope_high_degF = 5
            self.electricity_cooling_slope_high = convert_slope(electricity_cooling_slope_high_degF)

        if not self.heating_ref_temp_low < self.heating_ref_temp_x0 < self.heating_ref_temp_high:
            raise ValueError("Heating reference temperature parameter limits must be such that low < x0 < high")

        if not self.cooling_ref_temp_low < self.cooling_ref_temp_x0 < self.cooling_ref_temp_high:
            raise ValueError("Cooling reference temperature parameter limits must be such that low < x0 < high")

        if self.electricity_heating_slope_high < 0:
            raise ValueError("Electricity heating slope upper limit must be non-negative.")

        if self.natural_gas_heating_slope_high < 0:
            raise ValueError("Natural gas heating slope upper limit must be non-negative.")

        if self.electricity_cooling_slope_high < 0:
            raise ValueError("Electricity cooling slope upper limit must be non-negative.")

        meter_yaml = """
            !obj:eemeter.meter.Sequence {{
                sequence: [
                    !obj:eemeter.meter.BPI_2400_S_2012_ModelCalibrationUtilityBillCriteria {{
                        temperature_unit_str: {temp_unit}
                    }},
                    !obj:eemeter.meter.ForEachFuelType {{
                        fuel_types: [electricity, natural_gas],
                        fuel_unit_strs: [kWh, therms],
                        gathered_inputs: [
                            meets_model_calibration_utility_bill_criteria,
                        ],
                        meter: !obj:eemeter.meter.Condition {{
                            condition_parameter: meets_model_calibration_utility_bill_criteria_current_fuel,
                            success: !obj:eemeter.meter.Sequence &core_meter {{
                                sequence: [
                                    !obj:eemeter.meter.Switch {{
                                        target: fuel_type,
                                        cases: {{
                                            electricity: !obj:eemeter.meter.Sequence {{
                                                sequence: [
                                                    !obj:eemeter.meter.TemperatureSensitivityParameterOptimizationMeter {{
                                                        temperature_unit_str: {temp_unit},
                                                        model: !obj:eemeter.models.TemperatureSensitivityModel &electricity_model {{
                                                            cooling: True,
                                                            heating: True,
                                                            initial_params: {{
                                                                base_consumption: 0,
                                                                heating_slope: 0,
                                                                cooling_slope: 0,
                                                                heating_reference_temperature: {h_ref_x0},
                                                                cooling_reference_temperature: {c_ref_x0},
                                                            }},
                                                            param_bounds: {{
                                                                base_consumption: [-20,80],
                                                                heating_slope: [0,{e_h_slope_h}],
                                                                cooling_slope: [0,{e_c_slope_h}],
                                                                heating_reference_temperature: [{h_ref_l},{h_ref_h}],
                                                                cooling_reference_temperature: [{c_ref_l},{c_ref_h}],
                                                            }},
                                                        }},
                                                    }},
                                                    !obj:eemeter.meter.AnnualizedUsageMeter {{
                                                        temperature_unit_str: {temp_unit},
                                                        model: *electricity_model,
                                                    }},
                                                ]
                                            }},
                                            natural_gas: !obj:eemeter.meter.Sequence {{
                                                sequence: [
                                                    !obj:eemeter.meter.TemperatureSensitivityParameterOptimizationMeter {{
                                                        temperature_unit_str: {temp_unit},
                                                        model: !obj:eemeter.models.TemperatureSensitivityModel &natural_gas_model {{
                                                            cooling: False,
                                                            heating: True,
                                                            initial_params: {{
                                                                base_consumption: 0,
                                                                heating_slope: 0,
                                                                heating_reference_temperature: {h_ref_x0},
                                                            }},
                                                            param_bounds: {{
                                                                base_consumption: [-20,80],
                                                                heating_slope: [0,{n_g_h_slope_h}],
                                                                heating_reference_temperature: [{h_ref_l},{h_ref_h}],
                                                            }},
                                                        }},
                                                    }},
                                                    !obj:eemeter.meter.AnnualizedUsageMeter {{
                                                        temperature_unit_str: {temp_unit},
                                                        model: *natural_gas_model,
                                                    }},
                                                ]
                                            }},
                                        }},
                                    }},
                                    !obj:eemeter.meter.RMSE {{
                                        input_mapping: {{
                                            average_daily_usages: y,
                                            estimated_average_daily_usages: y_hat,
                                        }}
                                    }},
                                    !obj:eemeter.meter.RSquared {{
                                        input_mapping: {{
                                            average_daily_usages: y,
                                            estimated_average_daily_usages: y_hat,
                                        }}
                                    }},
                                ],
                            }},
                        }},
                    }},
                ]
            }}
            """.format(temp_unit=self.temperature_unit_str,
                       h_ref_l=self.heating_ref_temp_low,
                       h_ref_x0=self.heating_ref_temp_x0,
                       h_ref_h=self.heating_ref_temp_high,
                       e_h_slope_h=self.electricity_heating_slope_high,
                       n_g_h_slope_h=self.natural_gas_heating_slope_high,
                       c_ref_l=self.cooling_ref_temp_low,
                       c_ref_x0=self.cooling_ref_temp_x0,
                       c_ref_h=self.cooling_ref_temp_high,
                       e_c_slope_h=self.electricity_cooling_slope_high,
                       )
        return meter_yaml

    def evaluate_mapped_inputs(self,**kwargs):
        """PRISM-style evaluation of temperature sensitivity and
        weather-normalized annual consumption (NAC) at the single-project
        level.

        **Note:** In order to take advantage of input and output mappings, you
        should call the method :code:`meter.evaluate(**kwargs)` instead of
        this method.

        .. code-block:: python

            meter.evaluate(consumption_history=consumption_history,
                           weather_source=weather_source,
                           weather_normal_source=weather_normal_source)

        Parameters
        ----------
        consumption_history : eemeter.consumption.ConsumptionHistory
            All available consumption data for this project and reporting
            period.
        weather_source : eemeter.meter.WeatherSourceBase
            A weather source with data available for at least the duration of
            the reporting period.
        weather_normal_source : eemeter.meter.WeatherSourceBase
            A weather source which additionally provides the function
            :code:`weather_source.annual_daily_temperatures(unit)`.

        Returns
        -------
        out : dict
            The following results are always available:

            - *"average_daily_usages_bpi2400_electricity"* : Average electricity usage per
              day (kWh/day) for the electricity consumption periods.
            - *"average_daily_usages_bpi2400_natural_gas"* : Average natural gas usage per
              day (therms/day) for the natural gas consumption periods.
            - *"cdd_tmy"* : Total cooling degree days (base 65 degF or 18.33 degC)
              in a typical meteorological year (TMY3).
            - *"consumption_history_no_estimated"* : The input consumption history
              with estimated periods consolidated or removed.
            - *"cvrmse_electricity"* : The Coefficient of Variation of
              Root-mean-squared Error on the outputs of the electricity usage
              model.
            - *"cvrmse_natural_gas"* : The Coefficient of Variation of
              Root-mean-squared Error on the outputs of the electricity usage
              model.
            - *"electricity_presence"* : A boolean indicating presence of any
              electricity consumption data.
            - *"estimated_average_daily_usages_bpi2400_electricity"* : Average electricity
              usage per day (kWh/day) for the electricity consumption periods
              as estimated by the fitted temperature sensitivity model.
            - *"estimated_average_daily_usages_bpi2400_natural_gas"* : Average natural gas
              usage per day (therms/day) for the natural gas consumption
              periods as estimated by the fitted temperature sensitivity model.
            - *"has_enough_cdd_electricity"* : A boolean indicating whether or not
              the electricity consumption data covers (a) enough total CDD, (b)
              enough periods with low CDD, and (c) enough periods with high
              CDD.
            - *"has_enough_cdd_natural_gas"* : A boolean indicating whether or not
              the natural gas consumption data covers (a) enough total CDD, (b)
              enough periods with low CDD, and (c) enough periods with high
              CDD.
            - *"has_enough_data_electricity"* : A boolean indicating whether or not
              the electricity consumption data covers a period of at least 330
              days or a period of at least 184 days with enough CDD and HDD
              variation, as indicated by the results
              "has_enough_cdd_electricity" and "has_enough_hdd_electricity".
            - *"has_enough_data_natural_gas"* : A boolean indicating whether or not
              the natural gas consumption data covers a period of at least 330
              days or a period of at least 184 days with enough CDD and HDD
              variation, as indicated by the result
              "has_enough_hdd_cdd_natural_gas".
            - *"has_enough_hdd_cdd_electricity"* : A boolean indicating whether or
              not the electricity consumption data covers a period with enough
              variation in hdd and cdd; equivalent to the boolean value
              ("has_enough_cdd_electricity" and "has_enough_hdd_electricity")
            - *"has_enough_hdd_cdd_natural_gas"* : A boolean indicating whether or
              not the natural gas consumption data covers a period with enough
              variation in hdd and cdd; equivalent to the boolean value
              ("has_enough_cdd_natural_gas" and "has_enough_hdd_natural_gas")
            - *"has_enough_hdd_electricity"* : A boolean indicating whether or not
              the electricity consumption data covers (a) enough total HDD, (b)
              enough periods with low HDD, and (c) enough periods with high
              HDD.
            - *"has_enough_hdd_natural_gas"* : A boolean indicating whether or not
              the natural gas consumption data covers (a) enough total HDD, (b)
              enough periods with low HDD, and (c) enough periods with high
              HDD.
            - *"has_enough_periods_with_high_cdd_per_day_electricity"* : A boolean
              indicating whether or not the electricity consumption data has
              enough periods with at least 1.2x average normal CDD/day (TMY3,
              base 65 degF or 18.33 degC).
            - *"has_enough_periods_with_high_cdd_per_day_natural_gas"* : Always
              True.
            - *"has_enough_periods_with_high_hdd_per_day_electricity"* : A boolean
              indicating whether or not the electricity consumption data has
              enough periods with at least 1.2x average normal HDD/day (TMY3,
              base 65 degF or 18.33 degC).
            - *"has_enough_periods_with_high_hdd_per_day_natural_gas"* : A boolean
              indicating whether or not the natural_gas consumption data has
              enough periods with at least 1.2x average normal HDD/day (TMY3,
              base 65 degF or 18.33 degC).
            - *"has_enough_periods_with_low_cdd_per_day_electricity"* : A boolean
              indicating whether or not the electricity consumption data has
              enough periods with less than 0.2x average normal CDD/day (TMY3,
              base 65 degF or 18.33 degC).
            - *"has_enough_periods_with_low_cdd_per_day_natural_gas"* : Always
              True
            - *"has_enough_periods_with_low_hdd_per_day_electricity"* : A boolean
              indicating whether or not the electricity consumption data has
              enough periods with less than 0.2x average normal HDD/day (TMY3,
              base 65 degF or 18.33 degC).
            - *"has_enough_periods_with_low_hdd_per_day_natural_gas"* : A boolean
              indicating whether or not the natural gas consumption data has
              enough periods with less than 0.2x average normal HDD/day (TMY3,
              base 65 degF or 18.33 degC).
            - *"has_enough_total_cdd_electricity"* : A boolean indicating whether
              or not the total CDD during the total time span of the
              electricity data is at least 0.5x normal annual CDD/day (TMY3,
              base 65 degF or 18.33 degC).
            - *"has_enough_total_cdd_natural_gas"* : Always True
            - *"has_enough_total_hdd_electricity"* : A boolean indicating whether
              or not the total HDD during the total time span of the
              electricity data is at least 0.5x normal annual HDD/day (TMY3,
              base 65 degF or 18.33 degC).
            - *"has_enough_total_hdd_natural_gas"* : A boolean indicating whether
              or not the total CDD during the total time span of the
              natural gas data is at least 0.5x normal annual HDD/day (TMY3,
              base 65 degF or 18.33 degC).
            - *"has_recent_reading_electricity"* : A boolean indicating whether or
              not there is a electricity consumption data within 365 days of
              the current date (default) or the "since_date", the target end
              date of energy consumption evaluation period.
            - *"has_recent_reading_natural_gas"* : A boolean indicating whether or
              not there is a natural gas consumption data within 365 days of
              the current date (default) or the "since_date", the target end
              date of energy consumption evaluation period.
            - *"hdd_tmy"* : Total heating degree days (base 65 degF or 18.33 degC)
              in a typical meteorological year (TMY3).
            - *"meets_cvrmse_limit_electricity"* : A boolean indicating whether or
              not the Coefficient of Variation of the Root-mean-square Error
              (CVRMSE) of a regression of electricity consumption data against
              local observed HDD/CDD, as determined using equation 3.2.2.G.i
              of the ANSI/BPI-2400-S-2012 specification is less than 20.
            - *"meets_cvrmse_limit_natural_gas"* : A boolean indicating whether or
              not the Coefficient of Variation of the Root-mean-square Error
              (CVRMSE) of a regression of natural gas consumption data against
              local observed HDD/CDD, as determined using equation 3.2.2.G.i
              of the ANSI/BPI-2400-S-2012 specification is less than 20.
            - *"meets_model_calibration_utility_bill_criteria_electricity"* : A
              boolean indicating whether or not all electricity consumption
              acceptance criteria, as outlined in section 3.2.2 of the
              ANSI/BPI-2400-S-2012 specification, have been met.
            - *"meets_model_calibration_utility_bill_criteria_natural_gas"* : A
              boolean indicating whether or not all natural gas consumption
              acceptance criteria, as outlined in section 3.2.2 of the
              ANSI/BPI-2400-S-2012 specification, have been met.
            - *"n_periods_high_cdd_per_day_electricity"* : The number of
              electricity consumption data periods with observed CDD greater
              than 1.2x average normal CDD/day (TMY3, base 65 degF or 18.33
              degC).
            - *"n_periods_high_hdd_per_day_electricity"* : The number of
              electricity consumption data periods with observed HDD greater
              than 1.2x average normal CDD/day (TMY3, base 65 degF or 18.33
              degC).
            - *"n_periods_low_cdd_per_day_electricity"* : The number of
              electricity consumption data periods with observed CDD less than
              0.2x average normal CDD/day (TMY3, base 65 degF or 18.33 degC).
            - *"n_periods_low_hdd_per_day_electricity"* : The number of
              electricity consumption data periods with observed HDD less than
              1.2x average normal CDD/day (TMY3, base 65 degF or 18.33 degC).
            - *"n_periods_high_cdd_per_day_natural_gas"* : The number of natural
              gas consumption data periods with observed CDD greater than 1.2x
              average normal CDD/day (TMY3, base 65 degF or 18.33 degC).
            - *"n_periods_high_hdd_per_day_natural_gas"* : The number of natural
              gas consumption data periods with observed HDD greater than 1.2x
              average normal CDD/day (TMY3, base 65 degF or 18.33 degC).
            - *"n_periods_low_cdd_per_day_natural_gas"* : The number of natural gas
              consumption data periods with observed CDD less than 0.2x average
              normal CDD/day (TMY3, base 65 degF or 18.33 degC).
            - *"n_periods_low_hdd_per_day_natural_gas"* : The number of natural gas
              consumption data periods with observed HDD less than 1.2x average
              normal CDD/day (TMY3, base 65 degF or 18.33 degC).
            - *"natural_gas_presence"* : A boolean indicating presence of any
              natural_gas consumption data.
            - *"spans_183_days_and_has_enough_hdd_cdd_electricity"* : A boolean
              indicating whether or not electricity consumption data spans at
              least 184 days and is associated with sufficient breadth and
              variation in observed HDD and CDD.
            - *"spans_183_days_and_has_enough_hdd_cdd_natural_gas"* : A boolean
              indicating whether or not natural gas consumption data spans at
              least 184 days and is associated with sufficient breadth and
              variation in observed HDD and CDD.
            - *"spans_184_days_electricity"* : A boolean indicating whether or not
              electricity consumption data spans at least 184 days.
            - *"spans_184_days_natural_gas"* : A boolean indicating whether or not
              natural gas consumption data spans at least 184 days.
            - *"spans_330_days_electricity"* : A boolean indicating whether or not
              electricity consumption data spans at least 330 days.
            - *"spans_330_days_natural_gas"* : A boolean indicating whether or not
              natural gas consumption data spans at least 330 days.
            - *"temp_sensitivity_params_bpi2400_electricity"* : Fitted temperature
              sensitivity parameters for HDD/CDD electricity use model in an
              array of values with the following order: [base_consumption
              (kWh/day), heating_slope (kWh/HDD), heating_reference_temperature
              (degF or degC), cooling_slope (kWh/CDD),
              cooling_reference_temperature (degF or degC)].
            - *"temp_sensitivity_params_bpi2400_natural_gas"* : Fitted temperature
              sensitivity parameters for HDD/CDD natural gas use model in an
              array of values with the following order: [base_consumption
              (kWh/day), heating_slope (kWh/HDD), heating_reference_temperature
              (degF or degC)].
            - *"time_span_electricity"* : Number of days between earliest available
              electricity data and latest available electricity data.
            - *"time_span_natural_gas"* : Number of days between earliest available
              natural gas data and latest available natural gas data.
            - *"total_cdd_electricity"* : The total cooling degree days (base 65
              degF or 18.33 degC) observed during the all electricity
              consumption data periods.
            - *"total_cdd_natural_gas"* : The total cooling degree days (base 65
              degF or 18.33 degC) observed during the all natural gas
              consumption data periods.
            - *"total_hdd_electricity"* : The total heating degree days (base 65
              degF or 18.33 degC) observed during the all electricity
              consumption data periods.
            - *"total_hdd_natural_gas"* : The total heating degree days (base 65
              degF or 18.33 degC) observed during the all natural gas
              consumption data periods.

            The following results are only available if
            :code:`meets_model_calibration_utility_bill_criteria_electricity`
            is :code:`True`:

            - *"annualized_usage_electricity"* : Electricity usage in a typical
              meteorological year, as estimated by the fitted hdd/cdd
              electricity use model.
            - *"average_daily_usages_electricity"* : Average electricity usage per
              day (kWh/day) for the electricity consumption periods.
            - *"estimated_average_daily_usages_electricity"* : Average electricity
              usage per day (kWh/day) for the electricity consumption periods
              as estimated by the fitted temperature sensitivity model.
            - *"n_days_electricity"* : The number of days in each electricity
              consumption period; used as weights in model fitting.
            - *"rmse_electricity"* : Root-mean-square error of fitted hdd/cdd
              electricity use model estimations for all consumption periods.
            - *"r_squared_electricity"* : Coefficient of Determination (r^2) of
              fitted HDD/CDD electricity-use model estimations for all
              consumption periods.
            - *"temp_sensitivity_params_electricity"* : Fitted temperature
              sensitivity parameters for HDD/CDD electricity use model in an
              array of values with the following order: [base_consumption
              (kWh/day), heating_slope (kWh/HDD), heating_reference_temperature
              (degF or degC), cooling_slope (kWh/CDD),
              cooling_reference_temperature (degF or degC)]

            The following results are only available if
            :code:`meets_model_calibration_utility_bill_criteria_natural_gas`
            is :code:`True`:

            - *"annualized_usage_natural_gas"* : Natural gas usage in a typical
              meteorological year, as estimated by the fitted hdd/cdd
              natural gas use model.
            - *"average_daily_usages_natural_gas"* : Average natural gas usage per
              day (therms/day) for the natural gas consumption periods.
            - *"estimated_average_daily_usages_natural_gas"* : Average natural gas
              usage per day (therms/day) for the natural gas consumption
              periods as estimated by the fitted temperature sensitivity model.
            - *"n_days_natural_gas"* : The number of days in each natural_gas
              consumption period; used as weights in model fitting.
            - *"rmse_natural_gas"* : Root-mean-square error of fitted hdd/cdd
              natural gas use model estimations for all consumption periods.
            - *"r_squared_natural_gas"* : Coefficient of Determination (r^2) of
              fitted HDD/CDD natural gas use model estimations for all
              consumption periods.
            - *"temp_sensitivity_params_natural_gas"* : Fitted temperature
              sensitivity parameters for HDD/CDD natural gas use model in an
              array of values with the following order: [base_consumption
              (kWh/day), heating_slope (kWh/HDD), heating_reference_temperature
              (degF or degC)]

        """
        return self.meter.evaluate(**kwargs)

    def _get_child_inputs(self):
        return self.meter.get_inputs()
