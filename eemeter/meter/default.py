from eemeter.meter.base import MeterBase
from eemeter.config.yaml_parser import load

class DefaultResidentialMeter(MeterBase):
    """Implementation of the core EE-Meter savings evaluation method with
    defualt settings for evaluation of residential energy efficiency projects.

    Parameters
    ----------
    temperature_unit_str : {"degF", "degC"}, default "degC"
        Temperature unit to use throughout meter in degree-day calculations and
        parameter optimizations.
    electricity_baseload_low : float
        Lowest baseload in parameter optimization for electricity;
        defaults to 0 energy units/day
    electricity_baseload_x0 : float
        Initial baseload in parameter optimization for electricity;
        defaults to 0 energy units/day
    electricity_baseload_high : float
        Highest baseload in parameter optimization for electricity;
        defaults to 1000 energy units/day


    electricity_heating_slope_low : float
        Lowest heating slope in parameter optimization for electricity;
        defaults to 0 energy units/degF/day
    electricity_heating_slope_x0 : float
        Initial heating slope in parameter optimization for electricity;
        defaults to 0 energy units/degF/day
    electricity_heating_slope_high : float
        Highest heating slope in parameter optimization for electricity;
        defaults to or 100 energy units/degF/day

    electricity_cooling_slope_low : float
        Lowest cooling slope in parameter optimization for electricity;
        defaults to 0 energy units/degF/day
    electricity_cooling_slope_x0 : float
        Initial cooling slope in parameter optimization for electricity;
        defaults to 0 energy units/degF/day
    electricity_cooling_slope_high : float
        Highest cooling slope in parameter optimization for electricity;
        defaults to 100 energy units/degF/day


    natural_gas_baseload_low : float
        Lowest baseload in parameter optimization for natural gas;
        defaults to 0 energy units/day
    natural_gas_baseload_x0 : float
        Initial baseload in parameter optimization for natural gas;
        defaults to 0 energy units/day
    natural_gas_baseload_high : float
        Highest baseload in parameter optimization for natural gas;
        defaults to 1000 energy units/day

    natural_gas_heating_slope_low : float
        Lowest heating slope in parameter optimization for natural gas;
        defaults to 0 energy units/degF/day
    natural_gas_heating_slope_x0 : float
        Initial heating slope in parameter optimization for natural gas;
        defaults to 0 energy units/degF/day
    natural_gas_heating_slope_high : float
        Highest heating slope in parameter optimization for natural gas;
        defaults to 100 energy units/degF/day

    heating_ref_temp_low : float, default None
        Lowest heating reference temperature in parameter optimization;
        defaults to 55 degF
    heating_ref_temp_x0 : float
        Initial heating reference temperature in parameter optimization;
        defaults to 60 degF
    heating_ref_temp_high : float
        Highest heating reference temperature in parameter optimization;
        defaults to 70 degF
    cooling_ref_temp_low : float
        Lowest cooling reference temperature in parameter optimization;
        defaults to 60 degF
    cooling_ref_temp_x0 : float
        Initial cooling reference temperature in parameter optimization;
        defaults to 70 degF
    cooling_ref_temp_high : float
        Highest cooling reference temperature in parameter optimization;
        defaults to 75 degF
    """

    def __init__(self,temperature_unit_str="degC",
                      electricity_baseload_low=None,
                      electricity_baseload_x0=None,
                      electricity_baseload_high=None,
                      electricity_heating_slope_low=None,
                      electricity_heating_slope_x0=None,
                      electricity_heating_slope_high=None,
                      electricity_cooling_slope_low=None,
                      electricity_cooling_slope_x0=None,
                      electricity_cooling_slope_high=None,

                      natural_gas_baseload_low=None,
                      natural_gas_baseload_x0=None,
                      natural_gas_baseload_high=None,
                      natural_gas_heating_slope_low=None,
                      natural_gas_heating_slope_x0=None,
                      natural_gas_heating_slope_high=None,

                      heating_ref_temp_low=None,
                      heating_ref_temp_x0=None,
                      heating_ref_temp_high=None,
                      cooling_ref_temp_low=None,
                      cooling_ref_temp_x0=None,
                      cooling_ref_temp_high=None,
                      **kwargs):
        super(DefaultResidentialMeter, self).__init__(**kwargs)

        if temperature_unit_str not in ["degF","degC"]:
            raise ValueError("Invalid temperature_unit_str: should be one of 'degF' or 'degC'.")

        self.temperature_unit_str = temperature_unit_str

        self.electricity_baseload_high=electricity_baseload_high
        self.electricity_baseload_x0=electricity_baseload_x0
        self.electricity_baseload_low=electricity_baseload_low
        self.electricity_heating_slope_high=electricity_heating_slope_high
        self.electricity_heating_slope_x0=electricity_heating_slope_x0
        self.electricity_heating_slope_low=electricity_heating_slope_low
        self.electricity_cooling_slope_high=electricity_cooling_slope_high
        self.electricity_cooling_slope_low=electricity_cooling_slope_low
        self.electricity_cooling_slope_x0=electricity_cooling_slope_x0

        self.natural_gas_baseload_high=natural_gas_baseload_high
        self.natural_gas_baseload_low=natural_gas_baseload_low
        self.natural_gas_baseload_x0=natural_gas_baseload_x0
        self.natural_gas_heating_slope_high=natural_gas_heating_slope_high
        self.natural_gas_heating_slope_low=natural_gas_heating_slope_low
        self.natural_gas_heating_slope_x0=natural_gas_heating_slope_x0

        self.heating_ref_temp_low=heating_ref_temp_low
        self.heating_ref_temp_x0=heating_ref_temp_x0
        self.heating_ref_temp_high=heating_ref_temp_high
        self.cooling_ref_temp_low=cooling_ref_temp_low
        self.cooling_ref_temp_x0=cooling_ref_temp_x0
        self.cooling_ref_temp_high=cooling_ref_temp_high

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

        if self.electricity_baseload_low is None:
            electricity_baseload_low_degF = 0
            self.electricity_baseload_low = convert_slope(electricity_baseload_low_degF)
        if self.electricity_baseload_x0 is None:
            electricity_baseload_x0_degF = 0
            self.electricity_baseload_x0 = convert_slope(electricity_baseload_x0_degF)
        if self.electricity_baseload_high is None:
            electricity_baseload_high_degF = 1000
            self.electricity_baseload_high = convert_slope(electricity_baseload_high_degF)

        if self.electricity_heating_slope_low is None:
            electricity_heating_slope_low_degF = 0
            self.electricity_heating_slope_low = convert_slope(electricity_heating_slope_low_degF)
        if self.electricity_heating_slope_x0 is None:
            electricity_heating_slope_x0_degF = 0
            self.electricity_heating_slope_x0 = convert_slope(electricity_heating_slope_x0_degF)
        if self.electricity_heating_slope_high is None:
            electricity_heating_slope_high_degF = 100
            self.electricity_heating_slope_high = convert_slope(electricity_heating_slope_high_degF)

        if self.electricity_cooling_slope_low is None:
            electricity_cooling_slope_low_degF = 0
            self.electricity_cooling_slope_low = convert_slope(electricity_cooling_slope_low_degF)
        if self.electricity_cooling_slope_x0 is None:
            electricity_cooling_slope_x0_degF = 0
            self.electricity_cooling_slope_x0 = convert_slope(electricity_cooling_slope_x0_degF)
        if self.electricity_cooling_slope_high is None:
            electricity_cooling_slope_high_degF = 100
            self.electricity_cooling_slope_high = convert_slope(electricity_cooling_slope_high_degF)


        if self.natural_gas_baseload_low is None:
            natural_gas_baseload_low_degF = 0
            self.natural_gas_baseload_low = convert_slope(natural_gas_baseload_low_degF)
        if self.natural_gas_baseload_x0 is None:
            natural_gas_baseload_x0_degF = 0
            self.natural_gas_baseload_x0 = convert_slope(natural_gas_baseload_x0_degF)
        if self.natural_gas_baseload_high is None:
            natural_gas_baseload_high_degF = 1000
            self.natural_gas_baseload_high = convert_slope(natural_gas_baseload_high_degF)

        if self.natural_gas_heating_slope_low is None:
            natural_gas_heating_slope_low_degF = 0
            self.natural_gas_heating_slope_low = convert_slope(natural_gas_heating_slope_low_degF)
        if self.natural_gas_heating_slope_x0 is None:
            natural_gas_heating_slope_x0_degF = 0
            self.natural_gas_heating_slope_x0 = convert_slope(natural_gas_heating_slope_x0_degF)
        if self.natural_gas_heating_slope_high is None:
            natural_gas_heating_slope_high_degF = 100
            self.natural_gas_heating_slope_high = convert_slope(natural_gas_heating_slope_high_degF)


        if self.heating_ref_temp_low is None:
            heating_ref_temp_low_degF = 55
            self.heating_ref_temp_low = convert_temp(heating_ref_temp_low_degF)
        if self.heating_ref_temp_x0 is None:
            heating_ref_temp_x0_degF = 60
            self.heating_ref_temp_x0 = convert_temp(heating_ref_temp_x0_degF)
        if self.heating_ref_temp_high is None:
            heating_ref_temp_high_degF = 70
            self.heating_ref_temp_high = convert_temp(heating_ref_temp_high_degF)

        if self.cooling_ref_temp_low is None:
            cooling_ref_temp_low_degF = 60
            self.cooling_ref_temp_low = convert_temp(cooling_ref_temp_low_degF)
        if self.cooling_ref_temp_x0 is None:
            cooling_ref_temp_x0_degF = 70
            self.cooling_ref_temp_x0 = convert_temp(cooling_ref_temp_x0_degF)
        if self.cooling_ref_temp_high is None:
            cooling_ref_temp_high_degF = 75
            self.cooling_ref_temp_high = convert_temp(cooling_ref_temp_high_degF)


        if not 0 <= self.electricity_baseload_low <= \
                self.electricity_baseload_x0 <= self.electricity_baseload_high:
            message = "Electricity baseload parameter limits must be such " \
                    "that 0 <= low <= x0 <= high, but found low={}, x0={}, " \
                    "high={}".format(self.electricity_baseload_low,
                            self.electricity_baseload_x0,
                            self.electricity_baseload_high)
            raise ValueError(message)
        if not 0 <= self.electricity_heating_slope_low <= \
                self.electricity_heating_slope_x0 <= self.electricity_heating_slope_high:
            message = "Electricity heating slope parameter limits must be such " \
                    "that 0 <= low <= x0 <= high, but found low={}, x0={}, " \
                    "high={}".format(self.electricity_heating_slope_low,
                            self.electricity_heating_slope_x0,
                            self.electricity_heating_slope_high)
            raise ValueError(message)
        if not 0 <= self.electricity_cooling_slope_low <= \
                self.electricity_cooling_slope_x0 <= self.electricity_cooling_slope_high:
            message = "Electricity cooling slope parameter limits must be such " \
                    "that 0 <= low <= x0 <= high, but found low={}, x0={}, " \
                    "high={}".format(self.electricity_cooling_slope_low,
                            self.electricity_cooling_slope_x0,
                            self.electricity_cooling_slope_high)
            raise ValueError(message)

        if not 0 <= self.natural_gas_baseload_low <= \
                self.natural_gas_baseload_x0 <= self.natural_gas_baseload_high:
            message = "Natural gas baseload parameter limits must be such " \
                    "that 0 <= low <= x0 <= high, but found low={}, x0={}, " \
                    "high={}".format(self.natural_gas_baseload_low,
                            self.natural_gas_baseload_x0,
                            self.natural_gas_baseload_high)
            raise ValueError(message)
        if not 0 <= self.natural_gas_heating_slope_low <= \
                self.natural_gas_heating_slope_x0 <= self.natural_gas_heating_slope_high:
            message = "Natural gas heating slope parameter limits must be such " \
                    "that 0 <= low <= x0 <= high, but found low={}, x0={}, " \
                    "high={}".format(self.natural_gas_heating_slope_low,
                            self.natural_gas_heating_slope_x0,
                            self.natural_gas_heating_slope_high)
            raise ValueError(message)


        if not self.heating_ref_temp_low <= self.heating_ref_temp_x0 <= self.heating_ref_temp_high:
            raise ValueError("Heating reference temperature parameter limits must be such that low <= x0 <= high")
            message = "Heating reference temperature parameter limits must be such " \
                    "that low <= x0 <= high, but found low={}, x0={}, " \
                    "high={}".format(self.heating_ref_temp_low,
                            self.heating_ref_temp_x0,
                            self.heating_ref_temp_high)
            raise ValueError(message)
        if not self.cooling_ref_temp_low <= self.cooling_ref_temp_x0 <= self.cooling_ref_temp_high:
            message = "Cooling reference temperature parameter limits must be such " \
                    "that low <= x0 <= high, but found low={}, x0={}, " \
                    "high={}".format(self.cooling_ref_temp_low,
                            self.cooling_ref_temp_x0,
                            self.cooling_ref_temp_high)
            raise ValueError(message)

        meter_yaml = """
            !obj:eemeter.meter.Sequence {{
                sequence: [
                    !obj:eemeter.meter.ProjectAttributes {{
                        input_mapping: {{ project: {{}} }},
                        output_mapping: {{
                            weather_source: {{}},
                            weather_normal_source: {{}},
                        }},
                    }},
                    !obj:eemeter.meter.ProjectConsumptionDataBaselineReporting {{
                        input_mapping: {{ project: {{}} }},
                        output_mapping: {{ consumption: {{}} }}
                    }},
                    !obj:eemeter.meter.For {{
                        variable: {{ name: consumption_data }},
                        iterable: {{ name: consumption }},
                        meter: !obj:eemeter.meter.Sequence {{
                            sequence: [
                                !obj:eemeter.meter.BPI_2400_S_2012_ModelCalibrationUtilityBillCriteria {{
                                    temperature_unit_str: {temp_unit},
                                    tagspace: ["bpi2400"],
                                }},
                                !obj:eemeter.meter.Sequence {{
                                    sequence: [
                                        !obj:eemeter.meter.Switch {{
                                            target: {{
                                                name: fuel_type,
                                                tags: ["bpi2400"]
                                            }},
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
                                                                    base_consumption: [{e_bl_l},{e_bl_h}],
                                                                    heating_slope: [{e_h_slope_l},{e_h_slope_h}],
                                                                    cooling_slope: [{e_c_slope_l},{e_c_slope_h}],
                                                                    heating_reference_temperature: [{h_ref_l},{h_ref_h}],
                                                                    cooling_reference_temperature: [{c_ref_l},{c_ref_h}],
                                                                }},
                                                            }},
                                                            input_mapping: {{
                                                                consumption_data: {{}},
                                                                weather_source: {{}},
                                                                energy_unit_str: {{}},
                                                            }},
                                                            output_mapping: {{
                                                                temp_sensitivity_params: {{ name: model_params }},
                                                                average_daily_usages: {{}},
                                                                estimated_average_daily_usages: {{}},
                                                            }},
                                                        }},
                                                        !obj:eemeter.meter.AnnualizedUsageMeter {{
                                                            temperature_unit_str: {temp_unit},
                                                            model: *electricity_model,
                                                            input_mapping: {{
                                                                model_params: {{}},
                                                                weather_normal_source: {{}},
                                                            }},
                                                            output_mapping: {{
                                                                annualized_usage: {{}},
                                                            }},
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
                                                                    base_consumption: [{n_g_bl_l},{n_g_bl_h}],
                                                                    heating_slope: [{n_g_h_slope_l},{n_g_h_slope_h}],
                                                                    heating_reference_temperature: [{h_ref_l},{h_ref_h}],
                                                                }},
                                                            }},
                                                            input_mapping: {{
                                                                consumption_data: {{}},
                                                                weather_source: {{}},
                                                                energy_unit_str: {{}},
                                                            }},
                                                            output_mapping: {{
                                                                temp_sensitivity_params: {{ name: model_params }},
                                                                average_daily_usages: {{}},
                                                                estimated_average_daily_usages: {{}},
                                                            }},
                                                        }},
                                                        !obj:eemeter.meter.AnnualizedUsageMeter {{
                                                            temperature_unit_str: {temp_unit},
                                                            model: *natural_gas_model,
                                                            input_mapping: {{
                                                                model_params: {{}},
                                                                weather_normal_source: {{}},
                                                            }},
                                                            output_mapping: {{
                                                                annualized_usage: {{}},
                                                            }},
                                                        }},
                                                    ]
                                                }},
                                            }},
                                        }},
                                        !obj:eemeter.meter.RMSE {{
                                            input_mapping: {{
                                                y: {{ name: average_daily_usages }},
                                                y_hat: {{ name: estimated_average_daily_usages }},
                                            }},
                                            output_mapping: {{
                                                rmse: {{}},
                                            }}
                                        }},
                                        !obj:eemeter.meter.RSquared {{
                                            input_mapping: {{
                                                y: {{ name: average_daily_usages }},
                                                y_hat: {{ name: estimated_average_daily_usages }},
                                            }},
                                            output_mapping: {{
                                                r_squared: {{}},
                                            }}
                                        }},
                                    ],
                                }},
                            ]
                        }},
                    }},
                ]
            }}
            """.format(temp_unit=self.temperature_unit_str,
                       e_bl_h=self.electricity_baseload_high,
                       e_bl_x0=self.electricity_baseload_x0,
                       e_bl_l=self.electricity_baseload_low,
                       e_h_slope_h=self.electricity_heating_slope_high,
                       e_h_slope_x0=self.electricity_heating_slope_x0,
                       e_h_slope_l=self.electricity_heating_slope_low,
                       e_c_slope_h=self.electricity_cooling_slope_high,
                       e_c_slope_x0=self.electricity_cooling_slope_x0,
                       e_c_slope_l=self.electricity_cooling_slope_low,

                       n_g_bl_h=self.natural_gas_baseload_high,
                       n_g_bl_x0=self.natural_gas_baseload_x0,
                       n_g_bl_l=self.natural_gas_baseload_low,
                       n_g_h_slope_h=self.natural_gas_heating_slope_high,
                       n_g_h_slope_x0=self.natural_gas_heating_slope_x0,
                       n_g_h_slope_l=self.natural_gas_heating_slope_low,

                       h_ref_l=self.heating_ref_temp_low,
                       h_ref_x0=self.heating_ref_temp_x0,
                       h_ref_h=self.heating_ref_temp_high,
                       c_ref_l=self.cooling_ref_temp_low,
                       c_ref_x0=self.cooling_ref_temp_x0,
                       c_ref_h=self.cooling_ref_temp_high,

                       )
        return meter_yaml

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

              For electricty: [base_consumption (kWh/day), heating_slope
              (kWh/HDD), heating_reference_temperature (degF or degC),
              cooling_slope (kWh/CDD), cooling_reference_temperature (degF or
              degC)].

              For natural_gas: [base_consumption (kWh/day), heating_slope
              (kWh/HDD), heating_reference_temperature (degF or degC)].

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

              For electricty: [base_consumption (kWh/day), heating_slope
              (kWh/HDD), heating_reference_temperature (degF or degC),
              cooling_slope (kWh/CDD), cooling_reference_temperature (degF or
              degC)].

              For natural_gas: [base_consumption (kWh/day), heating_slope
              (kWh/HDD), heating_reference_temperature (degF or degC)].


        """
        return self.meter.evaluate(data_collection)

    def _get_child_inputs(self):
        return self.meter.get_inputs()
