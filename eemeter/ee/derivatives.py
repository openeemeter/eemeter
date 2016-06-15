import pandas as pd
from functools import wraps


# decorator that returns values as dicts
def derivative(output_name):
    def derivative_decorator(func):
        @wraps(func)
        def func_wrapper(*args, **kwargs):
            return {output_name: func(*args, **kwargs)}
        return func_wrapper
    return derivative_decorator


@derivative("annualized_weather_normal")
def annualized_weather_normal(energy_modeler, weather_normal_source):
    if energy_modeler is None:
        return None
    normal_index = pd.date_range('2015-01-01', freq='D', periods=365)
    normal_df = energy_modeler.create_demand_fixture(
            weather_normal_source, normal_index)
    normals = energy_modeler.predict(normal_df)
    annualized = normals.sum()
    n = energy_modeler.model.n
    upper = (energy_modeler.model.upper**2 * n)**0.5
    lower = (energy_modeler.model.lower**2 * n)**0.5
    return (annualized, (lower, upper), n)
