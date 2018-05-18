import pandas as pd
from eemeter.weather.noaa import ISDWeatherSource
from eemeter.weather.tmy3 import TMY3WeatherSource
from eemeter.weather.cz2010 import CZ2010WeatherSource
from datetime import datetime
import eeweather
import eemeter


def indexed_temperatures(usaf_id, index, unit,
    normalized, use_cz2010, allow_mixed_frequency=False):
    ''' Return average temperatures over the given index.

    Parameters
    ----------
    index : pandas.DatetimeIndex
        Index over which to supply average temperatures.
        The :code:`index` should be given as either an hourly ('H') or
        daily ('D') frequency.
    unit : str, {"degF", "degC"}
        Target temperature unit for returned temperature series.

    Returns
    -------
    temperatures : pandas.Series with DatetimeIndex
        Average temperatures over series indexed by :code:`index`.
    '''

    if index.shape == (0,):
        return pd.Series([], index=index, dtype=float)
    years = sorted(index.groupby(index.year).keys())
    start = pd.to_datetime(datetime(years[0],1,1), utc=True)
    end = pd.to_datetime(datetime(years[-1],12,31,23,59), utc=True)
    #using fully qualified name for monkeypatching
    tempC = eemeter.weather.eeweather_wrapper. \
        _get_temperature_data_eeweather(usaf_id, start, end,
        normalized, use_cz2010)

    if index.freq is not None:
        freq = index.freq
    else:
        try:
            freq = pd.infer_freq(index)
        except ValueError:
            freq = None

    if freq == 'D':
        return _daily_indexed_temperatures(index, unit, tempC)
    elif freq == 'H':
        return _hourly_indexed_temperatures(index, unit, tempC)
    elif allow_mixed_frequency:
        return _mixed_frequency_indexed_temperatures(index, unit, tempC)
    else:
        message = 'DatetimeIndex with unknown frequency not supported.'
        raise ValueError(message)

def _get_temperature_data_eeweather(usaf_id, start, end,
    normalized, use_cz2010):
    if normalized:
        if use_cz2010:
            tempC = eeweather.load_cz2010_hourly_temp_data(
                usaf_id, start, end)
        else:
            tempC = eeweather.load_tmy3_hourly_temp_data(
                usaf_id, start, end)
    else:
        tempC = eeweather.load_isd_hourly_temp_data(usaf_id, start, end)
    return tempC


def _daily_indexed_temperatures(index, unit, tempC):
    tempC = tempC.resample('D').mean()[index]
    return _unit_convert(tempC, unit)

def _hourly_indexed_temperatures(index, unit, tempC):
    message = (
        'DatetimeIndex frequency "H" not supported,'
        ' please resample to at least daily frequency ("D").'
        .format(index.freq)
    )
    raise ValueError(message)

def _mixed_frequency_indexed_temperatures(index, unit, tempC):
    min_period = _get_min_period(index)
    min_acceptable_period = _get_min_acceptable_period()

    if min_period < min_acceptable_period:
        message = (
            'DatetimeIndex with a period below "{}" (found: {}) not'
            ' supported.'
            .format(min_acceptable_period, min_period)
        )
        raise ValueError(message)

    index_ = _partitioned_multiindex(tempC.index, index)

    if index_ is None:
        message = 'Could not create partitioned mulitindex.'
        raise ValueError(message)

    level = index_.names[1]
    index_.get_level_values(level)
    values = tempC.reindex(index_.get_level_values(level)).values
    tempC = pd.DataFrame(values, index=index_)
    return _unit_convert(tempC, unit)

def _partitioned_multiindex(index_parts, index_periods, names=None):
    if names is None:
        if index_parts.freq == 'H':
            names = ["period", "hourly"]
        elif index_parts.freq == 'D':
            names = ["period", "daily"]
        else:
            message = (
                'Unexpected temperature frequency "{}".'
                .format(index_parts.freq)
            )
            raise ValueError(message)

    parts = iter(index_parts)
    periods = iter(index_periods)

    def _yield_index_tuples():
        part = next(parts, None)
        period_start = next(periods, None)
        period_end = next(periods, None)
        while part is not None:
            if period_start is None or period_end is None:
                break
            if part < period_start:
                part = next(parts, None)
            elif period_start <= part < period_end:
                yield (period_start, part)
                part = next(parts, None)
            else:
                period_start = period_end
                period_end = next(periods, None)

    index_tuples = list(_yield_index_tuples())
    if index_tuples == []:
        return None
    return pd.MultiIndex.from_tuples(index_tuples, names=names)

def _get_min_period(index):
    return index.to_series().diff().dropna().min()

def _get_min_acceptable_period():
    return pd.Timedelta('1 days')

def _unit_convert(x, unit):
    if unit is None or unit == "degC":
        return x
    elif unit == "degF":
        return 1.8 * x + 32
    else:
        message = (
            "Unit not supported ({}). Use 'degF' or 'degC'"
            .format(unit)
        )
        raise ValueError(message)
