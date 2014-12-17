from .base import MetricBase

import numpy as np
from scipy import stats

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

class TemperatureRegressionParametersMetric(MetricBase):

    # TODO - weight these by likelihood.
    balance_points = range(55,70)

    def __init__(self,unit_name,fuel_type):
        self.fuel_type = fuel_type

    def evaluate(self,consumption_history,weather_source):
        usages = [c.to("kWh") for c in consumption_history.get(self.fuel_type)]
        avg_temps = weather_source.get_average_temperature(consumption_history,self.fuel_type,"degF")
        best_coeffs = None,None
        best_r_value = -np.inf
        for balance_point in self.balance_points:
            u,t = self._filter_by_balance_point(balance_point,usages,avg_temps)
            slope, intercept, r_value, p_value, std_err = stats.linregress(u,t)
            if r_value > best_r_value and not np.isnan(p_value):
                best_coeffs = slope,intercept
        return best_coeffs

    @staticmethod
    def _filter_by_balance_point(balance_point,usages,avg_temps):
        data = [(usage,avg_temp) for usage,avg_temp in zip(usages,avg_temps) if avg_temp >= balance_point]
        if data:
            return zip(*data)
        else:
            return [],[]

class AverageTemperatureMetric(MetricBase):
    def __init__(self,fuel_type):
        self.fuel_type = fuel_type

    def evaluate(self,consumption_history,weather_source):
        avg_temps = weather_source.get_average_temperature(consumption_history,self.fuel_type,"degF")
        return np.mean(avg_temps)
