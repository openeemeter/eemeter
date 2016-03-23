import scipy.optimize as opt
import numpy as np
from .parameters import ParameterType
import inspect
import warnings

class BaseloadModelParameterType(ParameterType):
    parameters = [
        "base_daily_consumption"
    ]

class BaseloadHeatingModelParameterType(ParameterType):
    parameters = [
        "base_daily_consumption",
        "heating_balance_temperature",
        "heating_slope"
    ]

class BaseloadCoolingModelParameterType(ParameterType):
    parameters = [
        "base_daily_consumption",
        "cooling_balance_temperature",
        "cooling_slope"
    ]

class BaseloadHeatingCoolingModelParameterType(ParameterType):
    parameters = [
        "base_daily_consumption",
        "heating_balance_temperature",
        "heating_slope",
        "cooling_balance_temperature",
        "cooling_slope"
    ]

class Model(object):

    def __init__(self, initial_params=None, param_bounds=None, *args, **kwargs):
        if initial_params is None:
            self.initial_params = None
        else:
            self.initial_params = self.param_type(initial_params)

        if param_bounds is None:
            self.param_bounds = None
        else:
            self.param_bounds = self.param_type(param_bounds)

    def fit(self, X, y, weights=None):
        """Returns parameters which, according to an optimization routine in
        `scipy.optimize`, minimize the sum of squared errors between observed
        usages and the output of the a model which takes observed_daily_temps
        and returns usage estimates.
        """
        if self.initial_params is None:
            message = "must have initial_params defined for model fitting procedure."
            raise ValueError(message)
        else:
            x0 = self.initial_params.to_array()

        if self.param_bounds is None:
            bounds = None
        else:
            bounds = self.param_bounds.to_array()

        if weights is None:
            weights = 1

        def objective_function(param_array):
            y_est = self._transform(X, param_array)
            return np.nansum(((y - y_est)**2) * weights)


        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            result = opt.minimize(objective_function, x0=x0, bounds=bounds)
        params = result.x
        return self.param_type(params)

    def transform(self, X, params):
        return self._transform(X, params.to_array())

    def _transform(self, X, param_array):
        raise NotImplementedError

    def yaml_mapping(self):
        args = inspect.getargspec(self.__init__).args[1:]
        mapping = { arg: getattr(self,arg) for arg in args}
        mapping["initial_params"] = self.initial_params.to_dict()
        mapping["param_bounds"] = self.param_bounds.to_dict()
        return mapping

class AverageDailyTemperatureSensitivityModel(Model):

    def __init__(self, heating, cooling, *args, **kwargs):
        if cooling:
            if heating:
                self.model = AverageDailyBaseloadHeatingCoolingConsumptionModel(*args, **kwargs)
            else:
                self.model = AverageDailyBaseloadCoolingConsumptionModel(*args, **kwargs)
        else:
            if heating:
                self.model = AverageDailyBaseloadHeatingConsumptionModel(*args, **kwargs)
            else:
                self.model = AverageDailyBaseloadConsumptionModel(*args, **kwargs)
        self.cooling = cooling
        self.heating = heating
        self.param_type = self.model.param_type
        self.initial_params = self.model.initial_params
        self.param_bounds = self.model.param_bounds
        self._transform = self.model._transform

class AverageDailyBaseloadConsumptionModel(Model):

    def __init__(self, *args, **kwargs):
        self.param_type = BaseloadModelParameterType
        super(AverageDailyBaseloadConsumptionModel, self).__init__(*args, **kwargs)

    def _transform(self, X, param_array):
        base_daily_consumption = param_array[0]

        if not isinstance(X, np.ndarray):
            observed_daily_temps = np.array(X)
        else:
            observed_daily_temps = X

        return np.tile(base_daily_consumption, observed_daily_temps.shape[0])

class AverageDailyBaseloadHeatingConsumptionModel(Model):

    def __init__(self, *args, **kwargs):
        self.param_type = BaseloadHeatingModelParameterType
        super(AverageDailyBaseloadHeatingConsumptionModel, self).__init__(*args, **kwargs)

    def _transform(self, X, param_array):
        base_daily_consumption, heating_balance_temperature, heating_slope = \
                param_array

        if not isinstance(X, np.ndarray):
            observed_daily_temps = np.array(X)
        else:
            observed_daily_temps = X

        avg_daily_consumption_estimates = []
        for period_daily_temps in observed_daily_temps:
            if not isinstance(period_daily_temps, np.ndarray):
                period_daily_temps = np.array(period_daily_temps)

            daily_heating_demand = np.maximum(heating_balance_temperature - period_daily_temps, 0)
            avg_daily_heating_consumption = np.nanmean(daily_heating_demand * heating_slope)

            avg_daily_consumption_estimate = avg_daily_heating_consumption + base_daily_consumption
            avg_daily_consumption_estimates.append(avg_daily_consumption_estimate)

        return np.array(avg_daily_consumption_estimates)

class AverageDailyBaseloadCoolingConsumptionModel(Model):

    def __init__(self, *args, **kwargs):
        self.param_type = BaseloadCoolingModelParameterType
        super(AverageDailyBaseloadCoolingConsumptionModel, self).__init__(*args, **kwargs)

    def _transform(self, X, param_array):
        base_daily_consumption, cooling_balance_temperature, cooling_slope = \
                param_array

        if not isinstance(X, np.ndarray):
            observed_daily_temps = np.array(X)
        else:
            observed_daily_temps = X

        avg_daily_consumption_estimates = []
        for period_daily_temps in observed_daily_temps:
            if not isinstance(period_daily_temps, np.ndarray):
                period_daily_temps = np.array(period_daily_temps)

            daily_cooling_demand = np.maximum(period_daily_temps - cooling_balance_temperature, 0)
            avg_daily_cooling_consumption = np.nanmean(daily_cooling_demand * cooling_slope)

            avg_daily_consumption_estimate = avg_daily_cooling_consumption + base_daily_consumption
            avg_daily_consumption_estimates.append(avg_daily_consumption_estimate)

        return np.array(avg_daily_consumption_estimates)

class AverageDailyBaseloadHeatingCoolingConsumptionModel(Model):

    def __init__(self, *args, **kwargs):
        self.param_type = BaseloadHeatingCoolingModelParameterType
        super(AverageDailyBaseloadHeatingCoolingConsumptionModel, self).__init__(*args, **kwargs)

    def _transform(self, X, param_array):
        base_daily_consumption, heating_balance_temperature, heating_slope, \
                cooling_balance_temperature, cooling_slope = param_array

        if not isinstance(X, np.ndarray):
            observed_daily_temps = np.array(X)
        else:
            observed_daily_temps = X

        avg_daily_consumption_estimates = []
        for period_daily_temps in observed_daily_temps:
            if not isinstance(period_daily_temps, np.ndarray):
                period_daily_temps = np.array(period_daily_temps)

            daily_heating_demand = np.maximum(heating_balance_temperature - period_daily_temps, 0)
            avg_daily_heating_consumption = np.nanmean(daily_heating_demand * heating_slope)

            daily_cooling_demand = np.maximum(period_daily_temps - cooling_balance_temperature, 0)
            avg_daily_cooling_consumption = np.nanmean(daily_cooling_demand * cooling_slope)

            avg_daily_consumption_estimate = avg_daily_cooling_consumption + avg_daily_heating_consumption + base_daily_consumption
            avg_daily_consumption_estimates.append(avg_daily_consumption_estimate)
        return np.array(avg_daily_consumption_estimates)
