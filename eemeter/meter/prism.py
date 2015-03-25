from eemeter.meter.base import MeterBase
from eemeter.config.yaml_parser import load

class PRISMMeter(MeterBase):
    """Implementation of Princeton Scorekeeping Method.
    """

    def __init__(self,temperature_unit_str="degC",**kwargs):
        super(PRISMMeter, self).__init__(**kwargs)
        self.temperature_unit_str = temperature_unit_str
        self.meter = load(self._meter_yaml())

    def _meter_yaml(self):

        def degF_to_degC(F):
            return (F - 32.) * 5. / 9.

        heating_ref_temp_low_degF = 58
        heating_ref_temp_x0_degF = 60
        heating_ref_temp_high_degF = 66
        heating_slope_high_degF = 5

        cooling_ref_temp_low_degF = 64
        cooling_ref_temp_x0_degF = 70
        cooling_ref_temp_high_degF = 72
        cooling_slope_high_degF = 5

        heating_ref_temp_low_degC = degF_to_degC(heating_ref_temp_low_degF)
        heating_ref_temp_x0_degC = degF_to_degC(heating_ref_temp_x0_degF)
        heating_ref_temp_high_degC = degF_to_degC(heating_ref_temp_high_degF)
        heating_slope_high_degC = heating_slope_high_degF * 1.8

        cooling_ref_temp_low_degC = degF_to_degC(cooling_ref_temp_low_degF)
        cooling_ref_temp_x0_degC = degF_to_degC(cooling_ref_temp_x0_degF)
        cooling_ref_temp_high_degC = degF_to_degC(cooling_ref_temp_high_degF)
        cooling_slope_high_degC = cooling_slope_high_degF * 1.8

        if self.temperature_unit_str == "degF":
            heating_ref_temp_low = heating_ref_temp_low_degF
            heating_ref_temp_x0 = heating_ref_temp_x0_degF
            heating_ref_temp_high = heating_ref_temp_high_degF
            heating_slope_high = heating_slope_high_degF
            cooling_ref_temp_low = cooling_ref_temp_low_degF
            cooling_ref_temp_x0 = cooling_ref_temp_x0_degF
            cooling_ref_temp_high = cooling_ref_temp_high_degF
            cooling_slope_high = cooling_slope_high_degF
        elif self.temperature_unit_str == "degC":
            heating_ref_temp_low = heating_ref_temp_low_degC
            heating_ref_temp_x0 = heating_ref_temp_x0_degC
            heating_ref_temp_high = heating_ref_temp_high_degC
            heating_slope_high = heating_slope_high_degC
            cooling_ref_temp_low = cooling_ref_temp_low_degC
            cooling_ref_temp_x0 = cooling_ref_temp_x0_degC
            cooling_ref_temp_high = cooling_ref_temp_high_degC
            cooling_slope_high = heating_slope_high_degC
        else:
            raise ValueError("Invalid temperature_unit_str: should be one of 'degF' or 'degC'.")

        meter_yaml = """
            !obj:eemeter.meter.Sequence {{
                sequence: [
                    !obj:eemeter.meter.FuelTypePresenceMeter {{
                        fuel_types: [electricity,natural_gas]
                    }},
                    !obj:eemeter.meter.Condition {{
                        condition_parameter: electricity_presence,
                        success: !obj:eemeter.meter.Sequence {{
                            sequence: [
                                !obj:eemeter.meter.TemperatureSensitivityParameterOptimizationMeter {{
                                    fuel_unit_str: "kWh",
                                    fuel_type: "electricity",
                                    temperature_unit_str: "{temp_unit}",
                                    model: !obj:eemeter.models.TemperatureSensitivityModel &elec_model {{
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
                                            heating_slope: [0,{h_slope_h}],
                                            cooling_slope: [0,{c_slope_h}],
                                            heating_reference_temperature: [{h_ref_l},{h_ref_h}],
                                            cooling_reference_temperature: [{c_ref_l},{c_ref_h}],
                                        }},
                                    }},
                                }},
                                !obj:eemeter.meter.AnnualizedUsageMeter {{
                                    fuel_type: "electricity",
                                    temperature_unit_str: "{temp_unit}",
                                    model: *elec_model,
                                }},
                            ],
                            output_mapping: {{
                                temp_sensitivity_params: temp_sensitivity_params_electricity,
                                annualized_usage: annualized_usage_electricity,
                                daily_standard_error: daily_standard_error_electricity,
                            }},
                        }},
                    }},
                    !obj:eemeter.meter.Condition {{
                        condition_parameter: natural_gas_presence,
                        success: !obj:eemeter.meter.Sequence {{
                            sequence: [
                                !obj:eemeter.meter.TemperatureSensitivityParameterOptimizationMeter {{
                                    fuel_unit_str: "therms",
                                    fuel_type: "natural_gas",
                                    temperature_unit_str: "{temp_unit}",
                                    model: !obj:eemeter.models.TemperatureSensitivityModel &gas_model {{
                                        cooling: False,
                                        heating: True,
                                        initial_params: {{
                                            base_consumption: 0,
                                            heating_slope: 0,
                                            heating_reference_temperature: {h_ref_x0},
                                        }},
                                        param_bounds: {{
                                            base_consumption: [0,10],
                                            heating_slope: [0,{h_slope_h}],
                                            heating_reference_temperature: [{h_ref_l},{h_ref_h}],
                                        }},
                                    }},
                                }},
                                !obj:eemeter.meter.AnnualizedUsageMeter {{
                                    fuel_type: "natural_gas",
                                    temperature_unit_str: "{temp_unit}",
                                    model: *gas_model,
                                }},
                            ],
                            output_mapping: {{
                                temp_sensitivity_params: temp_sensitivity_params_natural_gas,
                                annualized_usage: annualized_usage_natural_gas,
                                daily_standard_error: daily_standard_error_natural_gas,
                            }},
                        }},
                    }},
                ]
            }}
            """.format(temp_unit=self.temperature_unit_str,
                       h_ref_l=heating_ref_temp_low,
                       h_ref_x0=heating_ref_temp_x0,
                       h_ref_h=heating_ref_temp_high,
                       h_slope_h=heating_slope_high,
                       c_ref_l=cooling_ref_temp_low,
                       c_ref_x0=cooling_ref_temp_x0,
                       c_ref_h=cooling_ref_temp_high,
                       c_slope_h=cooling_slope_high,
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
