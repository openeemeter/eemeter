from eemeter.meter.base import MeterBase
from eemeter.config.yaml_parser import load

class PRISMMeter(MeterBase):
    """Implementation of Princeton Scorekeeping Method.
    """

    def __init__(self,**kwargs):
        super(PRISMMeter, self).__init__(**kwargs)
        self.meter = load(self._meter_yaml())

    def _meter_yaml(self):
        meter_yaml = """
            !obj:eemeter.meter.Sequence {
                sequence: [
                    !obj:eemeter.meter.FuelTypePresenceMeter {
                        fuel_types: [electricity,natural_gas]
                    },
                    !obj:eemeter.meter.Condition {
                        condition_parameter: electricity_presence,
                        success: !obj:eemeter.meter.Sequence {
                            sequence: [
                                !obj:eemeter.meter.TemperatureSensitivityParameterOptimizationMeter {
                                    fuel_unit_str: "kWh",
                                    fuel_type: "electricity",
                                    temperature_unit_str: "degF",
                                    model: !obj:eemeter.models.TemperatureSensitivityModel &elec_model {
                                        cooling: True,
                                        heating: True,
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
                                daily_standard_error: daily_standard_error_electricity,
                            },
                        },
                    },
                    !obj:eemeter.meter.Condition {
                        condition_parameter: natural_gas_presence,
                        success: !obj:eemeter.meter.Sequence {
                            sequence: [
                                !obj:eemeter.meter.TemperatureSensitivityParameterOptimizationMeter {
                                    fuel_unit_str: "therms",
                                    fuel_type: "natural_gas",
                                    temperature_unit_str: "degF",
                                    model: !obj:eemeter.models.TemperatureSensitivityModel &gas_model {
                                        cooling: False,
                                        heating: True,
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
                                daily_standard_error: daily_standard_error_natural_gas,
                            },
                        },
                    },
                ]
            }
            """
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
            Dictionary of results like the following

            ::

                {
                    'electricity_presence': True,
                    'temp_sensitivity_params_electricity': array([  5.28680197e-02,   5.09216467e-02,   3.11451816e-01, 6.21000000e+01,   7.95406093e+00]),
                    'annualized_usage_electricity': 197.90877363035082,
                    'natural_gas_presence': True,
                    'temp_sensitivity_params_natural_gas': array([  6.36617250e+01,   2.64582535e-01,   2.89029866e-02]),
                    'annualized_usage_natural_gas': 133.05140290730517
                }

        """
        return self.meter.evaluate(**kwargs)

    def _get_child_inputs(self):
        return self.meter.get_inputs()
