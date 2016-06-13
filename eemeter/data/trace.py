import numpy as np

class EnergyTrace(object):
    """ Container for energy time series data.

    Parameters
    ----------
    fuel : str, {"electricity", "natural_gas"}
        The fuel type of the energy time series.
    interpretation : str
        The way this energy time series should be interpreted. Options are
        as follows:

        - `CONSUMPTION_SUPPLIED`: Amount of energy supplied by utility and
          used on site, not including locally generated electricity.
        - `CONSUMPTION_TOTAL`: Amount of energy consumed on site, including
          both supplied and locally generated energy, i.e.
          `CONSUMPTION_SUPPLIED + ON_SITE_GENERATION_CONSUMED`
        - `CONSUMPTION_NET`: Amount of supplied energy consumed on site minus
          amount of energy generated on site and fed back into the grid, i.e.
          `CONSUMPTION_SUPPLIED - ON_SITE_GENERATION_UNCONSUMED`
        - `ON_SITE_GENERATION_TOTAL`: Amount of locally generated energy
          including consumed and unconsumed energy, i.e.
          `ON_SITE_GENERATION_CONSUMED + ON_SITE_GENERATION_UNCONSUMED`
        - `ON_SITE_GENERATION_CONSUMED`: Amount of locally generated energy
          consumed on site.
        - `ON_SITE_GENERATION_UNCONSUMED`: Amount of excess locally generated
          energy fed back into the grid or sold back a utility.
        - `CIRCUIT`: Amount of energy consumed by a single circuit or plug
          load.

    data : pandas.DataFrame, default None
        Pandas DataFrame with two columns and a timezone-aware DatetimeIndex.
        Datetimes in the index are assumed to refer to the start of the period,
        and the value of the last datetime should be `NaN`, since is purpose is
        only to cap the end of the last period. (Other period ends are implied
        by next period starts.) DatetimeIndex does not need to have uniform
        frequency.

        - `value`: Amount of energy between this index and the next.
        - `estimated`: Whether or not the value was estimated. Particularly
          relevant for monthly billing data.

        If `serializer` instance is provided, this should be records in the
        format expected by the serializer.

    unit : str, {"kWh", "therm"}
        The name of the unit in which the energy time series is given.
    placeholder : bool
        Indicates that this EnergyTimeSeries is a placeholder - that for some
        reason it was unavailable, but its existence is still important.
    serializer : consumption.BaseSerializer
        Serializer instance to be used to deserialize records into a pandas
        dataframe. Must supply the `to_dataframe(records)` method.
    """

    # target_unit must be one of "kWh" or "therm"
    UNITS = {
        "kwh": {
            "target_unit": "kWh",
            "multiplier": 1.0,
        },
        "kWh": {
            "target_unit": "kWh",
            "multiplier": 1.0,
        },
        "KWH": {
            "target_unit": "kWh",
            "multiplier": 1.0,
        },
        "therm": {
            "target_unit": "therm",
            "multiplier": 1.0,
        },
        "therms": {
            "target_unit": "therm",
            "multiplier": 1.0,
        },
        "thm": {
            "target_unit": "therm",
            "multiplier": 1.0,
        },
        "THERM": {
            "target_unit": "therm",
            "multiplier": 1.0,
        },
        "THERMS": {
            "target_unit": "therm",
            "multiplier": 1.0,
        },
        "THM": {
            "target_unit": "therm",
            "multiplier": 1.0,
        },
        "wh": {
            "target_unit": "kWh",
            "multiplier": 0.001,
        },
        "Wh": {
            "target_unit": "kWh",
            "multiplier": 0.001,
        },
        "WH": {
            "target_unit": "kWh",
            "multiplier": 0.001,
        },
    }

    FUELS = [
        "electricity",
        "natural_gas",
    ]

    INTERPRETATIONS = [
        "CONSUMPTION_SUPPLIED",
        "CONSUMPTION_TOTAL",
        "CONSUMPTION_NET",
        "ON_SITE_GENERATION_TOTAL",
        "ON_SITE_GENERATION_CONSUMED",
        "ON_SITE_GENERATION_UNCONSUMED",
        "CIRCUIT",
    ]

    def __init__(self, fuel, interpretation, data=None, unit=None, placeholder=False, serializer=None):

        self._set_fuel(fuel)
        self._set_interpretation(interpretation)
        self._set_data(data, unit, placeholder, serializer)

    def _set_unit(self, unit):
        if unit in self.UNITS:
            self.unit = self.UNITS[unit]["target_unit"]
            self.unit_multiplier = self.UNITS[unit]["multiplier"]
        else:
            message = 'Unsupported unit: "{}".'.format(unit)
            raise ValueError(message)

    def _set_fuel(self, fuel):
        if fuel in self.FUELS:
            self.fuel = fuel
        else:
            message = 'Unsupported fuel: "{}".'.format(fuel)
            raise ValueError(message)

    def _set_interpretation(self, interpretation):
        if interpretation in self.INTERPRETATIONS:
            self.interpretation = interpretation
        else:
            message = 'Unsupported interpretation: "{}".'.format(fuel)
            raise ValueError(message)

    def _set_data(self, data, unit, placeholder, serializer):
        if data is None:
            if placeholder:
                self.data = None
                self.unit = None
                self.placeholder = True
            else:
                message = 'Supply `data` or set `placeholder=True`'
                raise ValueError(message)
        else:
            self._set_unit(unit)
            if not placeholder:
                if serializer is not None:
                    data = serializer.to_dataframe(data)
                self.data = data
                self.data.value = self.data.value * self.unit_multiplier
                self.placeholder = False
            else:
                message = 'Cannot have `placeholder=True` if data is supplied'
                raise ValueError(message)

    def __repr__(self):
        if self.placeholder:
            return (
                "EnergyTrace({}, {})\n  PLACEHOLDER"
                .format(self.fuel, self.interpretation)
            )
        else:
            return (
                "EnergyTrace({}, {})\n{}"
                .format(self.fuel, self.interpretation, self.data)
            )

    def filter_by_modeling_period(self, modeling_period):

        start = modeling_period.start_date
        end = modeling_period.end_date

        if start is None:
            if end is None:
                filtered_df = self.data.copy()
            else:
                filtered_df = self.data[:end].copy()
        else:
            if end is None:
                filtered_df = self.data[start:].copy()
            else:
                filtered_df = self.data[start:end].copy()

        # require NaN last data point as cap
        if filtered_df.shape[0] > 0:
            filtered_df.value.iloc[-1] = np.nan
            filtered_df.estimated.iloc[-1] = False

        return EnergyTrace(
            fuel=self.fuel,
            interpretation=self.interpretation,
            data=filtered_df,
            unit=self.unit
        )

    # def periods(self):
    #     """ Converts DatetimeIndex (which is timestamp based) to an list of
    #     Periods, which have associated start and endtimes.
    #
    #     Returns
    #     -------
    #     periods : list of eemeter.evaluation.Period
    #         A list of consumption periods.
    #     """
    #     if self.freq_timedelta is None:
    #         # ignore last period which is NaN and acting as an end point
    #         periods = [Period(start, end) for start, end in
    #                    zip(self.data.index,self.data.index[1:])]
    #         return periods
    #     else:
    #         periods = [Period(dt, dt + self.freq_timedelta)
    #                    for dt in self.data.index]
    #         return periods

    # def average_daily_consumptions(self):
    #     """ Computes average daily consumptions.
    #
    #     Returns
    #     -------
    #     averages : np.ndarray
    #         Array of average values in each period
    #     days : np.ndarray
    #         Array of number of days in each period
    #     """
    #     if self.freq_timedelta is None:
    #         # ignore last period which is NaN and acting as an end point
    #         avgs, n_days = [], []
    #         for v, ns in zip(self.data,np.diff(self.data.index)):
    #             # nanoseconds to days
    #             days = ns.astype('d')/8.64e13
    #             avgs.append(v/days)
    #             n_days.append(days)
    #         return np.array(avgs), np.array(n_days)
    #     else:
    #         days = self.freq_timedelta.days + self.freq_timedelta.seconds/8.64e4
    #         avgs, n_days = [], []
    #         for v in self.data:
    #             avgs.append(v/days)
    #             n_days.append(days)
    #         return np.array(avgs), np.array(n_days)

    # def total_period(self):
    #     """ The total period over which consumption data is recorded.
    #
    #     Returns
    #     -------
    #     period : eemeter.evaluation.Period
    #         The total time span covered by this ConsumptionData instance.
    #     """
    #     if self.data.shape[0] < 1:
    #         return None
    #     start_date = self.data.index[0]
    #     end_date = self.data.index[-1]
    #     if self.freq_timedelta is not None:
    #         end_date += self.freq_timedelta
    #     return Period(start_date, end_date)

    # def total_days(self):
    #     """ The total days over which consumption data is recorded.
    #
    #     Returns
    #     -------
    #     total_days : float
    #         The total days in the time span covered by this ConsumptionData
    #         instance.
    #     """
    #     period = self.total_period()
    #
    #     if period is None:
    #         return 0
    #     else:
    #         tdelta = period.timedelta
    #         return tdelta.days + tdelta.seconds/8.64e4

    # def json(self):
    #     return {
    #         "fuel_type": self.fuel_type,
    #         "unit_name": self.unit_name,
    #         "records": [{
    #             "start": r["start"].isoformat(),
    #             "end": r["end"].isoformat(),
    #             "value": r["value"],
    #             "estimated": r["estimated"],
    #         } for r in self.records()],
    #     }

    # def downsample(self, freq):
    #
    #     # empty case
    #     if self.data.shape[0] == 0:
    #         return copy.deepcopy(self)
    #
    #     rng = pd.date_range('2011-01-01', periods=2, freq=freq)
    #     target_period = rng[1] - rng[0]
    #
    #     index_series = pd.Series(self.data.index.tz_convert(pytz.UTC))
    #
    #     # are there any periods that would require a downsample?
    #     if index_series.shape[0] > 2:
    #         timedeltas = (index_series - index_series.shift())
    #
    #         for timedelta in timedeltas:
    #             if timedelta < target_period:
    #
    #                 # Found a short period. Need to resample.
    #                 consumption_resampled = ConsumptionData([],
    #                         self.fuel_type, self.unit_name,
    #                         record_type="arbitrary")
    #                 consumption_resampled.data = self.data.resample(freq).sum()
    #                 consumption_resampled.estimated = self.estimated.resample(freq).median().astype(bool)
    #                 return consumption_resampled
    #
    #     # Periods are all greater than or equal to downsample target, so just
    #     # return copy of self.
    #     return copy.deepcopy(self)
