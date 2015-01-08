import scipy.optimize as opt
import numpy as np

class ModelBase:
    pass

class DoubleBalancePointModel(ModelBase):
    def __init__(self,x0,bounds):
        self.x0 = x0
        self.bounds = bounds

    def parameter_optimization(self,usages,observed_temps):
        def objective_function(params):
            usages_est = self.compute_usage_estimates(params,observed_temps)
            return np.sum((usages - usages_est)**2)

        result = opt.minimize(objective_function,x0=self.x0,bounds=self.bounds)
        params = result.x
        return params

    @staticmethod
    def compute_usage_estimates(params,temps):
        # get parameters
        ts_low,ts_high,base_load,bp_low,bp_diff = params
        bp_high = bp_low + bp_diff

        estimates = []
        for temp in temps:
            estimate = base_load
            if temp <= bp_low:
                estimate += ts_low * (bp_low - temp)
            elif temp >= bp_high:
                estimate += ts_high * (temp - bp_high)
            estimates.append(estimate)

        return np.array(estimates)
