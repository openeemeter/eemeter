from eemeter.meter import MeterBase
from eemeter.config.yaml_parser import load

class BPI2400Meter(MeterBase):
    """Implementation of BPI-2400 standard
    """

    def __init__(self,**kwargs):
        super(BPI2400Meter, self).__init__(**kwargs)
        self.meter = load(self._meter_yaml())

    def _meter_yaml(self):
        meter_yaml = """
            !obj:eemeter.meter.Sequence {
                sequence: [
                    !obj:eemeter.meter.EstimatedReadingConsolidationMeter {
                    },
                    !obj:eemeter.meter.FuelTypePresenceMeter {
                        fuel_types: [electricity,natural_gas]
                    },
                    !obj:eemeter.meter.NormalAnnualHDD {
                        base: 65,
                        temperature_unit_str: degF,
                        output_mapping: {
                            normal_annual_hdd: hdd_65_tmy,
                        },
                    },
                    !obj:eemeter.meter.NormalAnnualCDD {
                        base: 65,
                        temperature_unit_str: degF,
                        output_mapping: {
                            normal_annual_cdd: cdd_65_tmy,
                        },
                    },
                    !obj:eemeter.meter.ForEachFuelType {
                        fuel_types: [electricity,natural_gas],
                        meter: !obj:eemeter.meter.Sequence {
                            input_mapping: {
                                consumption_history: null,
                                consumption_history_no_estimated: consumption_history
                            },
                            sequence: [
                                !obj:eemeter.meter.RecentReadingMeter {
                                    n_days: 360,
                                    output_mapping: {
                                        recent_reading: has_recent_reading
                                    }
                                },
                                !obj:eemeter.meter.TimeSpanMeter {
                                },
                                !obj:eemeter.meter.TotalHDDMeter {
                                    base: 65,
                                    temperature_unit_str: degF,
                                },
                                !obj:eemeter.meter.TotalCDDMeter {
                                    base: 65,
                                    temperature_unit_str: degF,
                                },
                                !obj:eemeter.meter.NPeriodsMeetingHDDPerDayThreshold {
                                    input_mapping: {
                                        hdd_65_tmy: hdd,
                                    },
                                    base: 65,
                                    temperature_unit_str: degF,
                                    operation: "gt",
                                    proportion: 0.0032876712,
                                    output_mapping: {
                                        n_periods: n_periods_high_hdd_per_day
                                    }
                                },
                                !obj:eemeter.meter.NPeriodsMeetingHDDPerDayThreshold {
                                    input_mapping: {
                                        hdd_65_tmy: hdd,
                                    },
                                    base: 65,
                                    temperature_unit_str: degF,
                                    operation: "lt",
                                    proportion: .00054794521,
                                    output_mapping: {
                                        n_periods: n_periods_low_hdd_per_day
                                    }
                                },
                                !obj:eemeter.meter.NPeriodsMeetingCDDPerDayThreshold {
                                    input_mapping: {
                                        cdd_65_tmy: cdd,
                                    },
                                    base: 65,
                                    temperature_unit_str: degF,
                                    operation: "gt",
                                    proportion: 0.0032876712,
                                    output_mapping: {
                                        n_periods: n_periods_high_cdd_per_day
                                    }
                                },
                                !obj:eemeter.meter.NPeriodsMeetingCDDPerDayThreshold {
                                    input_mapping: {
                                        cdd_65_tmy: cdd,
                                    },
                                    base: 65,
                                    temperature_unit_str: degF,
                                    operation: "lt",
                                    proportion: .00054794521,
                                    output_mapping: {
                                        n_periods: n_periods_low_cdd_per_day
                                    }
                                },
                                !obj:eemeter.meter.Switch {
                                    target: fuel_type,
                                    cases: {
                                        electricity: !obj:eemeter.meter.CVRMSE {
                                            fuel_unit_str: kWh,
                                            model: !obj:eemeter.models.TemperatureSensitivityModel {
                                                heating: True,
                                                cooling: True,
                                                initial_params: {
                                                    base_consumption: 0,
                                                    heating_slope: 0,
                                                    cooling_slope: 0,
                                                    heating_reference_temperature: 60,
                                                    cooling_reference_temperature: 70,
                                                },
                                                param_bounds: {
                                                    base_consumption: [-20,80],
                                                    heating_slope: [0,5],
                                                    cooling_slope: [0,5],
                                                    heating_reference_temperature: [58,66],
                                                    cooling_reference_temperature: [64,72],
                                                },
                                            }
                                        },
                                        natural_gas: !obj:eemeter.meter.CVRMSE {
                                            fuel_unit_str: therms,
                                            model: !obj:eemeter.models.TemperatureSensitivityModel {
                                                heating: true,
                                                cooling: false,
                                                initial_params: {
                                                    base_consumption: 0,
                                                    heating_slope: 0,
                                                    heating_reference_temperature: 60,
                                                },
                                                param_bounds: {
                                                    base_consumption: [0,10],
                                                    heating_slope: [0,5],
                                                    heating_reference_temperature: [58,66],
                                                },
                                            }
                                        },
                                    },
                                },
                                !obj:eemeter.meter.Switch {
                                    target: fuel_type,
                                    cases: {
                                        electricity: !obj:eemeter.meter.MeetsThresholds {
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
                                            thresholds: [330,183,hdd_65_tmy,cdd_65_tmy,1,1,1,1,20],
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
                                        },
                                        natural_gas: !obj:eemeter.meter.MeetsThresholds {
                                            values: [
                                                time_span,
                                                time_span,
                                                total_hdd,
                                                n_periods_high_hdd_per_day,
                                                n_periods_low_hdd_per_day,
                                                cvrmse,
                                            ],
                                            thresholds: [330,183,hdd_65_tmy,1,1,20],
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
                                            extras: {
                                                has_enough_total_cdd: true,
                                                has_enough_periods_with_high_cdd_per_day: true,
                                                has_enough_periods_with_low_cdd_per_day: true,
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
                                    output_mapping: {
                                        output: has_enough_hdd
                                    },
                                },
                                !obj:eemeter.meter.And {
                                    inputs: [
                                        has_enough_total_cdd,
                                        has_enough_periods_with_high_cdd_per_day,
                                        has_enough_periods_with_low_cdd_per_day,
                                    ],
                                    output_mapping: {
                                        output: has_enough_cdd
                                    },
                                },
                                !obj:eemeter.meter.And {
                                    inputs: [
                                        has_enough_hdd, has_enough_cdd
                                    ],
                                    output_mapping: {
                                        output: has_enough_hdd_cdd
                                    }
                                },
                                !obj:eemeter.meter.And {
                                    inputs: [
                                        spans_184_days, has_enough_hdd_cdd
                                    ],
                                    output_mapping: {
                                        output: spans_183_days_and_has_enough_hdd_cdd
                                    }
                                },
                                !obj:eemeter.meter.Or {
                                    inputs: [
                                        spans_330_days, spans_183_days_and_has_enough_hdd_cdd
                                    ],
                                    output_mapping: {
                                        output: has_enough_data
                                    }
                                },
                                !obj:eemeter.meter.And {
                                    inputs: [
                                        has_recent_reading, has_enough_data, meets_cvrmse_limit,
                                    ],
                                    output_mapping: {
                                        output: meets_model_calibration_utility_bill_criteria
                                    }
                                },
                                !obj:eemeter.meter.Condition {
                                    condition_parameter: meets_model_calibration_utility_bill_criteria,
                                    success: !obj:eemeter.meter.Sequence {
                                        input_mapping: {
                                            consumption_history: null,
                                            consumption_history_no_estimated: consumption_history
                                        },
                                        sequence: [
                                        ]
                                    },
                                    failure: !obj:eemeter.meter.Sequence {
                                        input_mapping: {
                                            consumption_history: null,
                                            consumption_history_no_estimated: consumption_history
                                        },
                                        sequence: [
                                        ]
                                    }
                                }
                            ]
                        }
                    },
                    !obj:eemeter.meter.Debug {},
                ]
            }
            """
        return meter_yaml

    def evaluate_mapped_inputs(self,**kwargs):
        return self.meter.evaluate(**kwargs)

    def _get_child_inputs(self):
        return self.meter.get_inputs()
