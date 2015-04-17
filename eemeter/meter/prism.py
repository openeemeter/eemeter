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
        defaults to 9 degC or 5 degF
    natural_gas_heating_slope_high : float
        Highest heating slope in parameter optimization for natural gas;
        defaults to 9 degC or 5 degF
    electricity_cooling_slope_high : float
        Highest cooling slope in parameter optimization for electricity;
        defaults to 9 degC or 5 degF
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
                    !obj:eemeter.meter.BPI_2400_S_2012_ModelCalibrationUtilityBillCriteria {{}},
                    !obj:eemeter.meter.ForEachFuelType {{
                        fuel_types: [electricity, natural_gas],
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
                                                        fuel_unit_str: kWh,
                                                        fuel_type: electricity,
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
                                                        fuel_type: electricity,
                                                        temperature_unit_str: {temp_unit},
                                                        model: *electricity_model,
                                                    }},
                                                ]
                                            }},
                                            natural_gas: !obj:eemeter.meter.Sequence {{
                                                sequence: [
                                                    !obj:eemeter.meter.TemperatureSensitivityParameterOptimizationMeter {{
                                                        fuel_unit_str: therms,
                                                        fuel_type: natural_gas,
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
                                                        fuel_type: natural_gas,
                                                        temperature_unit_str: {temp_unit},
                                                        model: *natural_gas_model,
                                                    }},
                                                ]
                                            }},
                                        }},
                                    }},
                                ],
                            }},
                            failure: *core_meter,
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
