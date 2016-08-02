from datetime import datetime

import pandas as pd
import pytz


def annualized_weather_normal(formatter, model, weather_normal_source):
    ''' Annualize energy trace values given a model and a source of 'normal'
    weather data, such as Typical Meteorological Year (TMY) 3 data.

    Parameters
    ----------
    formatter : eemeter.modeling.formatter.Formatter
        Formatter that can be used to create a demand fixure. Must supply the
        :code:`.create_demand_fixture(index, weather_source)` method.
    model : eemeter.modeling.models.Model
        Model that can be used to predict out of sample energy trace values.
        Must supply the :code:`.predict(demand_fixture_data)` method.
    weather_normal_source : eemeter.weather.WeatherSource
        WeatherSource providing weather normals.


    Returns
    -------
    out : dict
        Dictionary with the following item:

        - :code:`"annualized_weather_normal"`: 4-tuple with the values
          :code:`(annualized, lower, upper, n)`, where

          - :code:`annualized` is the total annualized (weather normalized)
            value predicted over the course of a 'normal' weather year.
          - :code:`lower` is the number which should be subtracted from
            :code:`annualized` to obtain the 0.025 quantile lower error bound.
          - :code:`upper` is the number which should be added to
            :code:`annualized` to obtain the 0.975 quantile upper error bound.
          - :code:`n` is the number of samples considered in developing the
            bound - useful for adding other values with errors.
    '''
    normal_index = pd.date_range('2015-01-01', freq='D', periods=365,
                                 tz=pytz.UTC)

    demand_fixture_data = formatter.create_demand_fixture(
            normal_index, weather_normal_source)

    normals = model.predict(demand_fixture_data)
    annualized = normals.sum()
    n = normal_index.shape[0]
    upper = (model.upper**2 * n)**0.5
    lower = (model.lower**2 * n)**0.5

    return {
        "annualized_weather_normal": (annualized, lower, upper, n),
    }


def gross_predicted(formatter, model, weather_source, reporting_period):
    ''' Find gross predicted energy trace values given a model and a source of
    observed weather data.

    Parameters
    ----------
    formatter : eemeter.modeling.formatter.Formatter
        Formatter that can be used to create a demand fixure. Must supply the
        :code:`.create_demand_fixture(index, weather_source)` method.
    model : eemeter.modeling.models.Model
        Model that can be used to predict out of sample energy trace values.
        Must supply the :code:`.predict(demand_fixture_data)` method.
    weather_source : eemeter.weather.WeatherSource
        WeatherSource providing observed weather data.
    baseline_period : eemeter.structures.ModelingPeriod
        Period targetted by baseline model.
    reporting_period : eemeter.structures.ModelingPeriod
        Period targetted by reporting model.

    Returns
    -------
    out : dict
        Dictionary with the following item:

        - :code:`"gross_predicted"`: 4-tuple with the values
          :code:`(annualized, lower, upper, n)`, where

          - :code:`gross_predicted` is the total gross predicted value
            over time period defined by the reporting period.
          - :code:`lower` is the number which should be subtracted from
            :code:`gross_predicted` to obtain the 0.025 quantile lower error
            bound.
          - :code:`upper` is the number which should be added to
            :code:`gross_predicted` to obtain the 0.975 quantile upper error
            bound.
          - :code:`n` is the number of samples considered in developing the
            bound - useful for adding other values with errors.
    '''
    start_date = reporting_period.start_date
    end_date = reporting_period.end_date
    if end_date is None:
        end_date = datetime.utcnow()
    index = pd.date_range(start_date, end_date, freq='D', tz=pytz.UTC)

    demand_fixture_data = formatter.create_demand_fixture(
        index, weather_source)

    values = model.predict(demand_fixture_data)
    gross_predicted = values.sum()
    n = index.shape[0]
    upper = (model.upper**2 * n)**0.5
    lower = (model.lower**2 * n)**0.5

    return {
        "gross_predicted": (gross_predicted, lower, upper, n),
    }
