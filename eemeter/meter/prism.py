from eemeter.meter.base import MeterBase
from eemeter.config.yaml_parser import load

class PRISMMeter(MeterBase):
    """Implementation of Princeton Scorekeeping Method.
    """

    def evaluate_mapped_inputs(self,**kwargs):
        meter_yaml = """
            !obj:eemeter.meter.SequentialMeter {
                sequence: [
                    !obj:eemeter.meter.FuelTypePresenceMeter {
                        fuel_types: [electricity,natural_gas]
                    },
                    !obj:eemeter.meter.ConditionalMeter {
                        condition_parameter: electricity_presence,
                        success: !obj:eemeter.meter.SequentialMeter {
                            sequence: [
                                !obj:eemeter.meter.TemperatureSensitivityParameterOptimizationMeter {
                                    fuel_unit_str: "kWh",
                                    fuel_type: "electricity",
                                    temperature_unit_str: "degF",
                                    model: !obj:eemeter.models.PRISMModel &elec_model {
                                        x0: [1.,1.,60],
                                        bounds: [[0,100],[0,100],[55,65]],
                                    },
                                },
                                !obj:eemeter.meter.AnnualizedUsageMeter {
                                    fuel_type: "electricity",
                                    temperature_unit_str: "degF",
                                    model: *elec_model,
                                },
                            ],
                            output_mapping: {
                                temp_sensitivity_params: temp_sensitivity_params_electricity,
                                annualized_usage: annualized_usage_electricity,
                            },
                        },
                    },
                    !obj:eemeter.meter.ConditionalMeter {
                        condition_parameter: natural_gas_presence,
                        success: !obj:eemeter.meter.SequentialMeter {
                            sequence: [
                                !obj:eemeter.meter.TemperatureSensitivityParameterOptimizationMeter {
                                    fuel_unit_str: "therms",
                                    fuel_type: "natural_gas",
                                    temperature_unit_str: "degF",
                                    model: !obj:eemeter.models.PRISMModel &gas_model {
                                        x0: [1.,1.,60],
                                        bounds: [[0,100],[0,100],[55,65]],
                                    },
                                },
                                !obj:eemeter.meter.AnnualizedUsageMeter {
                                    fuel_type: "natural_gas",
                                    temperature_unit_str: "degF",
                                    model: *gas_model,
                                },
                            ],
                            output_mapping: {
                                temp_sensitivity_params: temp_sensitivity_params_natural_gas,
                                annualized_usage: annualized_usage_natural_gas,
                            },
                        },
                    },
                ]
            }
            """
        meter = load(meter_yaml)
        return meter.evaluate(**kwargs)
