import numpy as np

def annualized_mean_usage(usages,unit):
    """
    Return annualized mean usage in the supplied unit
    """
    return np.mean([usage.to(unit) for usage in usages])
