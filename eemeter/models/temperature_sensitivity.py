import scipy.optimize as opt
import numpy as np

class ModelBase(object):
    def __init__(self,x0=None,bounds=None):
        self.x0 = x0
        self.bounds = bounds

    def parameter_optimization(self,average_daily_usages,observed_daily_temps, weights=None):
        """Returns parameters which, according to an optimization routine in
        `scipy.optimize`, minimize the sum of squared errors between observed
        usages and the output of the a model which takes observed_daily_temps
        and returns usage estimates.
        """
        # ignore nans
        average_daily_usages = np.ma.masked_array(average_daily_usages,np.isnan(average_daily_usages))

        # precalculate temps
        n_daily_temps = np.array([len(temps) for temps in observed_daily_temps])
        
        if weights == None:
            weights = np.ones(len(average_daily_usages))
        
        def objective_function(params):
            usages_est = self.compute_usage_estimates(params,observed_daily_temps)
            avg_usages_est = usages_est/n_daily_temps
            return np.sum(((average_daily_usages - avg_usages_est)**2)*weights)

        assert len(average_daily_usages) == len(observed_daily_temps)

        result = opt.minimize(objective_function,x0=self.x0,bounds=self.bounds)
        params = result.x
        return params

    @staticmethod
    def compute_usage_estimates(params,observed_daily_temps):
        """Return usage estimates given model parameters and temperature
        observations as an iterable of iterables containing daily temperatures
        (e.g. [[70.1,71.7,46.3],[10.1,15.1,16.0]]). Content and format of
        `params` must be determined by inheriting classes. Function must be
        implemented by all inheriting classes.
        """
        raise NotImplementedError

class HDDCDDBalancePointModel(ModelBase):

    @staticmethod
    def compute_usage_estimates(params,observed_daily_temps):
        """Returns usage estimates for a combined, dual balance point,
        heating/cooling degree day model. Parameters are given in the form
        `params = (ts_low, ts_high, base_load, bp_low, bp_diff)`, in which:

        - `ts_low` is the (generally positive) temperature sensitivity
          (units: usage per hdd) beyond the lower (heating degree day)
          balance point
        - `ts_high` is the (generally positive) temperature sensitivity
          (units: usage per cdd) beyond the upper (cooling degree day balance
          point
        - `base_load` is the daily non-temperature-related usage
        - `bp_low` is the reference temperature of the lower (hdd) balance
          point
        - `bd_diff` is the (generally positive) difference between the
          implicitly defined `bp_high` and `bp_low`

        """
        # get parameters
        ts_low,ts_high,base_load,bp_low,bp_diff = params
        bp_high = bp_low + bp_diff
        
        result = []
        for interval_daily_temps in observed_daily_temps:
            cooling = np.maximum(interval_daily_temps - bp_high, 0)*ts_high
            heating = np.maximum(bp_low - interval_daily_temps, 0)*ts_low
            total = np.sum(cooling+heating) + base_load*len(interval_daily_temps)
            result.append(total)
        return result

class HDDBalancePointModel(ModelBase):

    @staticmethod
    def compute_usage_estimates(params,observed_daily_temps):
        """Returns usage estimates for a simple single balance point, heating
        degree day model. Parameters are given in the form
        `params = (reference_temperature, base_level_consumption,
        heating_slope)`, in which:

        - `reference_temperature` is the temperature base of the heating
          degree day balance point
        - `base_level_consumption` is the daily non-temperature-related
          usage
        - `heating_slope` is the (generally positive) temperature
          sensitivity (units: usage per hdd) beyond the heating degree day
          reference temperature

        """
        # get parameters
        reference_temperature,base_level_consumption,heating_slope = params
        
        result = []
        for interval_daily_temps in observed_daily_temps:
            heating = np.maximum(reference_temperature - interval_daily_temps, 0)*heating_slope
            total = np.sum(heating) + base_level_consumption*len(interval_daily_temps)
            result.append(total)
        return result

class CDDBalancePointModel(ModelBase):

    @staticmethod
    def compute_usage_estimates(params,observed_daily_temps):
        """Returns usage estimates for a simple single balance point, heating
        degree day model. Parameters are given in the form
        `params = (reference_temperature, base_level_consumption,
        cooling_slope)`, in which:

        - `reference_temperature` is the temperature base of the cooling
          degree day balance point
        - `base_level_consumption` is the daily non-temperature-related
          usage
        - `cooling_slope` is the (generally positive) temperature
          sensitivity (units: usage per cdd) beyond the cooling degree day
          reference temperature

        """
        # get parameters
        reference_temperature,base_level_consumption,cooling_slope = params

        result = []
        for interval_daily_temps in observed_daily_temps:
            cooling = np.maximum(interval_daily_temps - reference_temperature, 0)*cooling_slope
            total = np.sum(cooling) + base_level_consumption*len(interval_daily_temps)
            result.append(total)
        return result
