from eemeter.meter import MeterBase
from eemeter.config.yaml_parser import load
from datetime import datetime
import pytz

class BPI_2400_S_2012_ModelCalibrationUtilityBillCriteria(MeterBase):
    """Implementation of BPI-2400-S-2012 section 3.2.2.
    """

    def __init__(self,temperature_unit_str,**kwargs):
        super(BPI_2400_S_2012_ModelCalibrationUtilityBillCriteria, self).__init__(**kwargs)
        self.temperature_unit_str = temperature_unit_str
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

        heating_ref_temp_low = convert_temp(58)
        heating_ref_temp_x0 = convert_temp(60)
        heating_ref_temp_high = convert_temp(66)
        hdd_base = convert_temp(65)
        cooling_ref_temp_low = convert_temp(64)
        cooling_ref_temp_x0 = convert_temp(70)
        cooling_ref_temp_high = convert_temp(72)
        cdd_base = convert_temp(65)
        electricity_heating_slope_high = convert_slope(5)
        natural_gas_heating_slope_high = convert_slope(5)
        electricity_cooling_slope_high = convert_slope(5)

        meter_yaml = """
            !obj:eemeter.meter.Sequence {{
                sequence: [
                    !obj:eemeter.meter.EstimatedReadingConsolidationMeter {{
                    }},
                    !obj:eemeter.meter.FuelTypePresenceMeter {{
                        fuel_types: [electricity,natural_gas]
                    }},
                    !obj:eemeter.meter.NormalAnnualHDD {{
                        base: {hdd_base},
                        temperature_unit_str: {temp_unit},
                        output_mapping: {{
                            normal_annual_hdd: hdd_tmy,
                        }},
                    }},
                    !obj:eemeter.meter.NormalAnnualCDD {{
                        base: {cdd_base},
                        temperature_unit_str: {temp_unit},
                        output_mapping: {{
                            normal_annual_cdd: cdd_tmy,
                        }},
                    }},
                    !obj:eemeter.meter.ForEachFuelType {{
                        fuel_types: [electricity,natural_gas],
                        fuel_unit_strs: [kWh,therms],
                        meter: !obj:eemeter.meter.Sequence {{
                            input_mapping: {{
                                consumption_history: null,
                                consumption_history_no_estimated: consumption_history
                            }},
                            sequence: [
                                !obj:eemeter.meter.RecentReadingMeter {{
                                    n_days: 360,
                                    output_mapping: {{
                                        recent_reading: has_recent_reading
                                    }}
                                }},
                                !obj:eemeter.meter.TimeSpanMeter {{
                                }},
                                !obj:eemeter.meter.TotalHDDMeter {{
                                    base: {hdd_base},
                                    temperature_unit_str: {temp_unit},
                                }},
                                !obj:eemeter.meter.TotalCDDMeter {{
                                    base: {cdd_base},
                                    temperature_unit_str: {temp_unit},
                                }},
                                !obj:eemeter.meter.NPeriodsMeetingHDDPerDayThreshold {{
                                    input_mapping: {{
                                        hdd_tmy: hdd,
                                    }},
                                    base: {hdd_base},
                                    temperature_unit_str: {temp_unit},
                                    operation: "gt",
                                    proportion: 0.0032876712,
                                    output_mapping: {{
                                        n_periods: n_periods_high_hdd_per_day
                                    }}
                                }},
                                !obj:eemeter.meter.NPeriodsMeetingHDDPerDayThreshold {{
                                    input_mapping: {{
                                        hdd_tmy: hdd,
                                    }},
                                    base: {hdd_base},
                                    temperature_unit_str: {temp_unit},
                                    operation: "lt",
                                    proportion: .00054794521,
                                    output_mapping: {{
                                        n_periods: n_periods_low_hdd_per_day
                                    }}
                                }},
                                !obj:eemeter.meter.NPeriodsMeetingCDDPerDayThreshold {{
                                    input_mapping: {{
                                        cdd_tmy: cdd,
                                    }},
                                    base: {cdd_base},
                                    temperature_unit_str: {temp_unit},
                                    operation: "gt",
                                    proportion: 0.0032876712,
                                    output_mapping: {{
                                        n_periods: n_periods_high_cdd_per_day
                                    }}
                                }},
                                !obj:eemeter.meter.NPeriodsMeetingCDDPerDayThreshold {{
                                    input_mapping: {{
                                        cdd_tmy: cdd,
                                    }},
                                    base: {cdd_base},
                                    temperature_unit_str: {temp_unit},
                                    operation: "lt",
                                    proportion: .00054794521,
                                    output_mapping: {{
                                        n_periods: n_periods_low_cdd_per_day
                                    }}
                                }},
                                !obj:eemeter.meter.Switch {{
                                    target: fuel_type,
                                    cases: {{
                                        electricity: !obj:eemeter.meter.TemperatureSensitivityParameterOptimizationMeter {{
                                            temperature_unit_str: {temp_unit},
                                            model: !obj:eemeter.models.TemperatureSensitivityModel {{
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
                                        natural_gas: !obj:eemeter.meter.TemperatureSensitivityParameterOptimizationMeter {{
                                            temperature_unit_str: {temp_unit},
                                            model: !obj:eemeter.models.TemperatureSensitivityModel {{
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
                                    }},
                                    output_mapping: {{
                                        average_daily_usages: average_daily_usages_bpi2400,
                                        estimated_average_daily_usages: estimated_average_daily_usages_bpi2400,
                                        temp_sensitivity_params: temp_sensitivity_params_bpi2400,
                                        daily_standard_error: null,
                                        n_days: null,
                                    }},
                                }},
                                !obj:eemeter.meter.CVRMSE {{
                                    input_mapping: {{
                                        average_daily_usages_bpi2400: y,
                                        estimated_average_daily_usages_bpi2400: y_hat,
                                        temp_sensitivity_params_bpi2400: params,
                                    }},
                                }},
                                !obj:eemeter.meter.Switch {{
                                    target: fuel_type,
                                    cases: {{
                                        electricity: !obj:eemeter.meter.MeetsThresholds {{
                                            values: [
                                                time_span,
                                                time_span,
                                                total_hdd,
                                                total_cdd,
                                                n_periods_high_hdd_per_day,
                                                n_periods_low_hdd_per_day,
                                                n_periods_high_cdd_per_day,
                                                n_periods_low_cdd_per_day,
                                                cvrmse,
                                            ],
                                            thresholds: [330,183,hdd_tmy,cdd_tmy,1,1,1,1,20],
                                            operations: [gte,gt,gt,gt,gte,gte,gte,gte,lte],
                                            proportions: [1,1,.5,.5,1,1,1,1,1],
                                            output_names: [
                                                spans_330_days,
                                                spans_184_days,
                                                has_enough_total_hdd,
                                                has_enough_total_cdd,
                                                has_enough_periods_with_high_hdd_per_day,
                                                has_enough_periods_with_low_hdd_per_day,
                                                has_enough_periods_with_high_cdd_per_day,
                                                has_enough_periods_with_low_cdd_per_day,
                                                meets_cvrmse_limit,
                                            ],
                                        }},
                                        natural_gas: !obj:eemeter.meter.MeetsThresholds {{
                                            values: [
                                                time_span,
                                                time_span,
                                                total_hdd,
                                                n_periods_high_hdd_per_day,
                                                n_periods_low_hdd_per_day,
                                                cvrmse,
                                            ],
                                            thresholds: [330,183,hdd_tmy,1,1,20],
                                            operations: [gte,gt,gt,gte,gte,lte],
                                            proportions: [1,1,.5,1,1,1],
                                            output_names: [
                                                spans_330_days,
                                                spans_184_days,
                                                has_enough_total_hdd,
                                                has_enough_periods_with_high_hdd_per_day,
                                                has_enough_periods_with_low_hdd_per_day,
                                                meets_cvrmse_limit,
                                            ],
                                            extras: {{
                                                has_enough_total_cdd: true,
                                                has_enough_periods_with_high_cdd_per_day: true,
                                                has_enough_periods_with_low_cdd_per_day: true,
                                            }},
                                        }},
                                    }}
                                }},
                                !obj:eemeter.meter.And {{
                                    inputs: [
                                        has_enough_total_hdd,
                                        has_enough_periods_with_high_hdd_per_day,
                                        has_enough_periods_with_low_hdd_per_day,
                                    ],
                                    output_mapping: {{
                                        output: has_enough_hdd
                                    }},
                                }},
                                !obj:eemeter.meter.And {{
                                    inputs: [
                                        has_enough_total_cdd,
                                        has_enough_periods_with_high_cdd_per_day,
                                        has_enough_periods_with_low_cdd_per_day,
                                    ],
                                    output_mapping: {{
                                        output: has_enough_cdd
                                    }},
                                }},
                                !obj:eemeter.meter.And {{
                                    inputs: [
                                        has_enough_hdd, has_enough_cdd
                                    ],
                                    output_mapping: {{
                                        output: has_enough_hdd_cdd
                                    }}
                                }},
                                !obj:eemeter.meter.And {{
                                    inputs: [
                                        spans_184_days, has_enough_hdd_cdd
                                    ],
                                    output_mapping: {{
                                        output: spans_183_days_and_has_enough_hdd_cdd
                                    }}
                                }},
                                !obj:eemeter.meter.Or {{
                                    inputs: [
                                        spans_330_days, spans_183_days_and_has_enough_hdd_cdd
                                    ],
                                    output_mapping: {{
                                        output: has_enough_data
                                    }}
                                }},
                                !obj:eemeter.meter.And {{
                                    inputs: [
                                        has_recent_reading, has_enough_data, meets_cvrmse_limit,
                                    ],
                                    output_mapping: {{
                                        output: meets_model_calibration_utility_bill_criteria
                                    }}
                                }},
                            ],
                        }}
                    }}
                ]
            }}
            """.format(temp_unit=self.temperature_unit_str,
                       h_ref_l=heating_ref_temp_low,
                       h_ref_x0=heating_ref_temp_x0,
                       h_ref_h=heating_ref_temp_high,
                       c_ref_l=cooling_ref_temp_low,
                       c_ref_x0=cooling_ref_temp_x0,
                       c_ref_h=cooling_ref_temp_high,
                       e_h_slope_h=electricity_heating_slope_high,
                       n_g_h_slope_h=natural_gas_heating_slope_high,
                       e_c_slope_h=electricity_cooling_slope_high,
                       hdd_base=hdd_base,
                       cdd_base=cdd_base,
                       )
        return meter_yaml

    def evaluate_mapped_inputs(self,**kwargs):
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
        out : dict

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

        """
        return self.meter.evaluate(**kwargs)

    def _get_child_inputs(self):
        return self.meter.get_inputs()
