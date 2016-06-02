from datetime import timedelta

import numpy as np
import pandas as pd
from pandas.core.common import is_list_like

class WeatherSourceBase(object):

    def __init__(self, station):
        self.station = station
        self.tempC = pd.Series(dtype=float)

    @staticmethod
    def _unit_convert(x, unit):
        if unit is None or unit == "degC":
            return x
        elif unit == "degF":
            return 1.8*x + 32
        else:
            message = "Unit not supported ({}). Use 'degF' or 'degC'".format(unit)
            raise NotImplementedError(message)

    def _fetch_period(self, period):
        message = "Inheriting classes must override this method."
        raise NotImplementedError(message)

    def _fetch_datetime(self, dt):
        message = "Inheriting classes must override this method."
        raise NotImplementedError(message)

    def average_temperature(self, periods, unit):
        """The average temperatures during each period as calculated by taking
        the mean of all available daily average temperatures during that
        period.

        Parameters
        ----------
        periods : [list of] eemeter.evaluation.Period
            Time periods over which temperatures will be aggregated. A single
            datetime period may be given.
        unit : {"degC", "degF"}
            The unit in which average temperatures should be returned.

        Returns
        -------
        out : np.ndarray
            Array of average temperatures observed during each period. If a
            single datetime period is given, a single temperature will be
            returned as a float.
        """

        if is_list_like(periods):
            values = np.array([self._period_average_temperature(p, None) for p in periods])
            return self._unit_convert(values, unit)
        else:
            return self._period_average_temperature(periods, unit)

    def daily_temperatures(self, periods, unit):
        """The daily average temperatures for each period.

        Parameters
        ----------
        periods : [list of] eemeter.evaluation.Period
            Time periods over which temperatures will be aggregated. A single
            datetime period may be given.
        unit : {"degC", "degF"}
            The unit in which temperatures should be returned.

        Returns
        -------
        out : np.ndarray
            Array of arrays of average daily temperatures observed during each
            period. Note: array is not guaranteed to be rectangular. If a
            single datetime period is given, a single numpy array of
            temperatures will be returned.
        """

        if is_list_like(periods):
            values = np.array([self._period_daily_temperatures(p, None) for p in periods])
            return self._unit_convert(values, unit)
        else:
            return self._period_daily_temperatures(periods, unit)

    def hourly_temperatures(self, periods, unit):
        """The hourly observed temperatures for each period.

        Parameters
        ----------
        periods : [list of] eemeter.evaluation.Period
            Time periods over which temperatures will be collected. A single
            datetime period may be given.
        unit : {"degC", "degF"}
            The unit in which temperatures should be returned.

        Returns
        -------
        out : np.ndarray
            Array of arrays of observed_daily temperatures observed during each
            period. Note: array is not guaranteed to be rectangular. If a
            single datetime period is given, a single numpy array of
            temperatures will be returned.
        """

        if is_list_like(periods):
            values = np.array([self._period_hourly_temperatures(p, None) for p in periods])
            return self._unit_convert(values, unit)
        else:
            return self._period_hourly_temperatures(periods, unit)

    def _period_average_temperature(self, period, unit):
        self._fetch_period(period)
        value = self.tempC[period.start:period.end - timedelta(seconds=1)].mean()
        return self._unit_convert(value, unit)

    def _period_daily_temperatures(self, period, unit):
        self._fetch_period(period)
        temps = []
        for days in range(period.timedelta.days):
            dt = period.start + timedelta(days=days)
            temps.append(self.datetime_average_temperature(dt, unit))
        return np.array(temps)

    def _period_hourly_temperatures(self, period, unit):
        self._fetch_period(period)
        temps = []
        for seconds in range(0, int(period.timedelta.total_seconds()), 3600):
            dt = period.start + timedelta(seconds=seconds)
            temps.append(self.datetime_hourly_temperature(dt, unit))
        return np.array(temps)

    def datetime_average_temperature(self, dt, unit):
        """The daily average temperatures for a particular period.

        Parameters
        ----------
        dt : datetime.datetime or datetime.date
            The date for which to find or calculate the average temperature.
        unit : {"degC", "degF"}
            The unit in which the daily average temperature should be returned.

        Returns
        -------
        out : np.ndarray
            Average temperature observed.
        """

        try:
            value = self.tempC[dt.strftime("%Y-%m-%d")].mean()
            return self._unit_convert(value, unit)
        except KeyError:
            try:
                self._fetch_datetime(dt)
                value = self.tempC[dt.strftime("%Y-%m-%d")].mean()
                return self._unit_convert(value, unit)
            except KeyError:
                pass
        return np.nan

    def datetime_hourly_temperature(self, dt, unit):
        """The hourly observed temperatures for each period.

        Parameters
        ----------
        periods : [list of] eemeter.evaluation.Period
            Time periods over which temperatures will be collected. A single
            datetime period may be given.
        unit : {"degC", "degF"}
            The unit in which temperatures should be returned.

        Returns
        -------
        out : np.ndarray
            Array of arrays of observed_daily temperatures observed during each
            period. Note: array is not guaranteed to be rectangular. If a
            single datetime period is given, a single numpy array of
            temperatures will be returned.
        """

        try:
            return self._unit_convert(self.tempC[dt], unit)
        except KeyError:
            try:
                self._fetch_datetime(dt)
                return self._unit_convert(self.tempC[dt], unit)
            except KeyError:
                pass
        return np.nan

    def hdd(self, periods, unit, base, per_day=False):
        """The total heating degree days observed during each time period.

        Parameters
        ----------
        periods : [list of] eemeter.evaluation.Period
            Time periods over which heating degree days will be calculated and
            collected. A single period may be supplied.
        unit : {"degC", "degF"}
            The temperature unit to be used
        base : int or float
            The base of the heating degree day
        per_day : bool
            If True, the total should be returned as an average instead of
            a sum.

        Returns
        -------
        out : np.ndarray
            Array of heating degree days observed during each time period.
        """

        if is_list_like(periods):
            return np.array([self._period_hdd(p, unit, base, per_day) for p in periods])
        else:
            return self._period_hdd(periods, unit, base, per_day)

    def cdd(self, periods, unit, base, per_day=False):
        """The total cooling degree days observed during each time period.

        Parameters
        ----------
        periods : list of eemeter.evaluation.Period objects
            Time periods over which cooling degree days will be calculated and
            collected. A single Period may be given.
        unit : {"degC", "degF"}
            The temperature unit to be used
        base : int or float
            The base of the cooling degree day
        per_day : bool, default=False
            If True, the total should be returned as an average instead of
            a sum.

        Returns
        -------
        out : np.ndarray
            Array of cooling degree days observed during each time period.
        """

        if is_list_like(periods):
            return np.array([self._period_cdd(p, unit, base, per_day) for p in periods])
        else:
            return self._period_cdd(periods, unit, base, per_day)

    def _period_hdd(self, period, unit, base, per_day):
        """The total heating degree days observed during a particular
        time period.

        Parameters
        ----------
        period : eemeter.evaluation.Period
            Time period over which heating degree days will be summed.
        unit : {"degC", "degF"}
            The temperature unit to be used
        base : int or float
            The base of the heating degree day
        per_day : bool
            If True, the total should be returned as an average instead of
            a sum.

        Returns
        -------
        out : float
            Total heating degree days observed during the time period.
        """

        temps = self._period_daily_temperatures(period,unit)
        masked_temps = np.ma.masked_array(temps, np.isnan(temps))
        total_hdd = np.sum(np.maximum(base - masked_temps,0))
        if per_day:
            n_days = period.timedelta.days
            return total_hdd / n_days
        else:
            return total_hdd

    def _period_cdd(self, period, unit, base, per_day):
        """The total cooling degree days observed during a particular
        time period.

        Parameters
        ----------
        period : eemeter.evaluation.Period
            Time period over which cooling degree days will be summed.
        unit : {"degC", "degF"}
            The temperature unit to be used
        base : int or float
            The base of the cooling degree day
        per_day : bool
            If True, the total should be returned as an average instead of
            a sum.

        Returns
        -------
        out : float
            Total cooling degree days observed during the time period.
        """

        temps = self._period_daily_temperatures(period,unit)
        masked_temps = np.ma.masked_array(temps, np.isnan(temps))
        total_cdd = np.sum(np.maximum(masked_temps - base,0))
        if per_day:
            n_days = period.timedelta.days
            return total_cdd / n_days
        else:
            return total_cdd

    def json(self):
        return {
            "station": self.station,
            "records": [{
                "datetime": d.strftime(self.cache_date_format),
                "tempC": t if pd.notnull(t) else None,
            } for d, t in self.tempC.iteritems()]
        }

