import scipy.optimize as opt
import numpy as np

class TemperatureSensitivityModel(object):
    def __init__(self,cooling,heating,initial_params=None,param_bounds=None):
        """Parameters (and bounds) are given in the following form. If bounds,
        each value should be a tuple or list of length 2, with upper and lower
        bounds; if initial params, each value should be an integer or a float.
        `initial_params = {"base_consumption":...,
                   "heating_slope":...,
                   "heating_reference_temperature":...
                   "cooling_slope":...
                   "cooling_reference_temperature":...}`, in which:
        - `base_consumption` is the daily non-temperature-related usage
        - `heating_slope` is the reference temperature of the lower (hdd)
          balance point
        - `heating_reference_temperature` is the (generally positive)
          temperature sensitivity (units: usage per hdd) beyond the lower
          (hdd) balance point
        - `cooling_slope` is the reference temperature of the higher (cdd)
          balance point
        - `cooling_reference_temperature` is the (generally positive)
          temperature sensitivity (units: usage per cdd) beyond the upper
          (cdd) balance point
        """
        self.cooling = cooling
        self.heating = heating
        self.initial_params = initial_params
        self.param_bounds = param_bounds

    def parameter_optimization(self,average_daily_usages,observed_daily_temps,weights=None):
        """Returns parameters which, according to an optimization routine in
        `scipy.optimize`, minimize the sum of squared errors between observed
        usages and the output of the a model which takes observed_daily_temps
        and returns usage estimates.
        """
        # ignore nans
        average_daily_usages = np.ma.masked_array(average_daily_usages,np.isnan(average_daily_usages))

        # precalculate temps
        n_daily_temps = np.array([len(temps) for temps in observed_daily_temps])

        if weights is None:
            weights = np.ones(len(average_daily_usages))

        def objective_function(params):
            usages_est = self.compute_usage_estimates(params,observed_daily_temps)
            avg_usages_est = usages_est/n_daily_temps
            return np.sum(((average_daily_usages - avg_usages_est)**2)*weights)

        assert len(average_daily_usages) == len(observed_daily_temps)

        x0 = self.param_dict_to_list(self.initial_params)
        bounds = self.param_dict_to_list(self.param_bounds)
        result = opt.minimize(objective_function,x0=x0,bounds=bounds)
        params = result.x
        return params

    def param_dict_to_list(self,params):
        """Return a list of parameters given a parameter dictionary
        """
        param_list = [params["base_consumption"]]
        if self.heating:
            param_list.append(params["heating_slope"])
            param_list.append(params["heating_reference_temperature"])
        if self.cooling:
            param_list.append(params["cooling_slope"])
            param_list.append(params["cooling_reference_temperature"])
        return param_list

    def compute_usage_estimates(self,params,observed_daily_temps):
        """Returns usage estimates for a combined, dual balance point,
        heating/cooling degree day model. Expects params to be of the form
        created by `param_dict_to_list`.
        """
        # get parameters
        base_consumption = params[0]
        if self.heating:
            heating_slope, heating_reference_temperature = params[1:3]
        else:
            heating_slope, heating_reference_temperature = None,None
        
        if self.cooling:
            cooling_params = params[3:5] if self.heating else params[1:3]
            cooling_slope, cooling_reference_temperature = cooling_params
        else:
            cooling_slope, cooling_reference_temperature = None,None
        
        usage_estimates = []
        for interval_daily_temps in observed_daily_temps:
            if not isinstance(interval_daily_temps, np.ndarray):
                interval_daily_temps = np.array(interval_daily_temps)
            if self.cooling:
                cooling = np.maximum(interval_daily_temps - cooling_reference_temperature, 0)*cooling_slope
            else:
                cooling = 0
            if self.heating:
                heating = np.maximum(heating_reference_temperature - interval_daily_temps, 0)*heating_slope
            else:
                heating = 0
            total_usage = np.sum(cooling + heating) + base_consumption*interval_daily_temps.shape[0]
            usage_estimates.append(total_usage)
        return np.array(usage_estimates)
