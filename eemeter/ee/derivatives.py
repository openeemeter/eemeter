import pandas as pd
import pytz


def annualized_weather_normal(formatter, model, weather_normal_source):
    normal_index = pd.date_range('2015-01-01', freq='D', periods=365,
                                 tz=pytz.UTC)

    normal_df = formatter.create_demand_fixture(
            normal_index, weather_normal_source)

    normals = model.predict(normal_df)
    annualized = normals.sum()
    n = model.n
    upper = (model.upper**2 * n)**0.5
    lower = (model.lower**2 * n)**0.5

    return {
        "annualized_weather_normal": (annualized, lower, upper, n),
    }
