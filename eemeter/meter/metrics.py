from .base import MetricBase

import numpy as np
from scipy import stats
import scipy.optimize as opt

class RawAverageUsageMetric(MetricBase):
    def __init__(self,unit_name,fuel_type=None):
        # TODO - allow different units for different fuel types.
        self.unit_name = unit_name
        super(RawAverageUsageMetric,self).__init__(fuel_type)

    def evaluate_fuel_type(self,consumptions):
        """Returns the average usage with the specified unit and the specified
        fuel type.
        """
        if consumptions is None:
            return np.nan
        return np.mean([consumption.to(self.unit_name) for consumption in consumptions])

class AverageTemperatureMetric(MetricBase):
    def __init__(self,fuel_type):
        self.fuel_type = fuel_type

    def evaluate(self,consumption_history,weather_source):
        consumptions = consumption_history.get(self.fuel_type)
        avg_temps = weather_source.get_average_temperature(consumptions,"degF")
        return np.mean(avg_temps)

class HDDCDDTemperatureSensitivityParametersMetric(MetricBase):
    def __init__(self,unit_name,fuel_type):
        self.fuel_type = fuel_type
        self.unit_name = unit_name
        self.temperature_unit_name = "degF" # TODO - unhardcode this

    def evaluate(self,consumption_history,weather_source):
        consumptions = consumption_history.get(self.fuel_type)
        usages = [c.to(self.unit_name) for c in consumptions]
        observed_temps = weather_source.get_average_temperature(consumptions,self.temperature_unit_name)
        params = self._parameter_optimization(usages,observed_temps)
        return params

    @staticmethod
    def _parameter_optimization(usages,observed_temps):
        def _objective_function(params):
            # get parameters
            ts_low,ts_high,base_load,bp_low,bp_diff = params
            bp_high = bp_low + bp_diff

            # split by balance point
            usage_low_temp,usage_mid_temp,usage_high_temp = [],[],[]
            temp_low,temp_mid,temp_high = [],[],[]
            for usage,temp in zip(usages,observed_temps):
                if temp <= bp_low:
                    usage_low_temp.append(usage)
                    temp_low.append(temp)
                elif temp >= bp_high:
                    usage_high_temp.append(usage)
                    temp_high.append(temp)
                else:
                    usage_mid_temp.append(usage)
                    temp_mid.append(temp)

            usage_low_temp = np.array(usage_low_temp)
            usage_mid_temp = np.array(usage_mid_temp)
            usage_high_temp = np.array(usage_high_temp)
            temp_low = np.array(temp_low)
            temp_mid = np.array(temp_mid)
            temp_high = np.array(temp_high)

            usage_est_low_temp = ts_low * (temp_low - bp_low) + base_load
            usage_est_mid_temp = (0 * temp_mid) + base_load
            usage_est_high_temp = ts_high * (temp_high - bp_high) + base_load

            low = usage_est_low_temp - usage_low_temp
            mid = usage_est_mid_temp - usage_mid_temp
            high = usage_est_high_temp - usage_high_temp

            squares = np.concatenate((low,mid,high))**2
            return np.sum(squares)

        x0 = [-10,1.,1.,60,8]
        bounds = [(-200,0),(0,200),(0,2000),(55,65),(5,12)]

        result = opt.minimize(_objective_function,x0,bounds=bounds)
        params = result.x
        return params

class WeatherNormalizedAverageUsageMetric(MetricBase):
    def __init__(self,unit_name,fuel_type):
        self.fuel_type = fuel_type
        self.unit_name = unit_name
        self.temperature_unit_name = "degF" # TODO - unhardcode this

    def evaluate(self,consumption_history,temperature_sensitivity_parameters,weather_normal_source):
        consumptions = consumption_history.get(self.fuel_type)
        normal_temps = weather_normal_source.get_daily_temperatures(self.temperature_unit_name)
        return np.sum(self._estimate_usages(normal_temps,temperature_sensitivity_parameters))

    @staticmethod
    def _estimate_usages(temps,params):
        ts_low,ts_high,base_load,bp_low,bp_diff = params
        bp_high = bp_low + bp_diff
        estimates = []
        for temp in temps:
            usage = base_load
            if temp >= bp_high:
                usage += ts_high * (temp - bp_high)
            elif temp <= bp_low:
                usage += ts_low * (temp - bp_low)
            estimates.append(usage)
        return estimates

class TotalHDDMetric(MetricBase):
    def __init__(self,fuel_type):
        self.fuel_type = fuel_type
        self.temperature_unit_name = "degF" # TODO - unhardcode this

    def evaluate(self,consumption_history,temperature_sensitivity_parameters,weather_source):
        consumptions = consumption_history.get(self.fuel_type)
        base = temperature_sensitivity_parameters[3]
        hdd_per_month = weather_source.get_hdd(consumptions,self.temperature_unit_name,base)
        return np.sum(hdd_per_month)

class TotalCDDMetric(MetricBase):
    def __init__(self,fuel_type):
        self.fuel_type = fuel_type
        self.temperature_unit_name = "degF" # TODO - unhardcode this

    def evaluate(self,consumption_history,temperature_sensitivity_parameters,weather_source):
        consumptions = consumption_history.get(self.fuel_type)
        base = temperature_sensitivity_parameters[3] + temperature_sensitivity_parameters[4]
        cdd_per_month = weather_source.get_cdd(consumptions,self.temperature_unit_name,base)
        return np.sum(cdd_per_month)
