import numpy as np

def annualized_mean_usage(usages,energy_unit):
    """
    Return annualized mean usage in the supplied unit
    """
    return np.mean([usage.to(energy_unit) for usage in usages])

def weather_normalize(usages,temperature_unit,normals):
    pass
