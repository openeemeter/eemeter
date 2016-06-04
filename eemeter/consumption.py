from . import ureg, Q_

from datetime import datetime
import pandas as pd
import numpy as np
import warnings
import pytz
import copy

from eemeter.evaluation import Period

class BaseSerializer(object):

    sort_key = None
    required_fields = []
    datetime_fields = []

    def _sort_records(self, records):
        if self.sort_key is None:
            message = (
                'Must supply cls.sort_key in class definition.'
            )
            raise AttributeError(message)

        try:
            sorted_records = sorted(records, key=lambda x: x[self.sort_key])
        except KeyError:
            message = (
                'Sorting failed due to missing key {} in record.'
                .format(self.sort_key)
            )
            raise KeyError(message)

        return sorted_records

    def _validated_tuples_to_dataframe(self, validated_tuples):

        if validated_tuples == []:
            dts, values, estimateds = [], [], []
        else:
            dts, values, estimateds = zip(*validated_tuples)

        df = pd.DataFrame(
            {"value": values, "estimated": estimateds},
            index=pd.DatetimeIndex(dts),
            columns=["value", "estimated"],
        )
        df.value = df.value.astype(float)
        df.estimated = df.estimated.astype(bool)
        return df

    def to_dataframe(self, records):
        """
        Returns a dataframe of records.
        """
        sorted_records = self._sort_records(records)
        validated_tuples = list(self.validate_records(sorted_records))
        return self._validated_tuples_to_dataframe(validated_tuples)

    def validate_records(self, sorted_records):
        """
        Yields validated (start (datetime), value (float), estimated (bool))
        tuples of data.
        """
        raise NotImplementedError('`validate_records()` must be implemented.')

    def validate_record(self, record):

        # make sure required fields are available
        for field in self.required_fields:
            if field not in record:
                message = (
                    'Record missing "{}" field:\n{}'
                    .format(field, record)
                )
                raise ValueError(message)

        # make sure dates/datetimes are tz aware
        for field in self.datetime_fields:
            dt = record[field]
            if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
                message = (
                    'Record field ("{}": {}) is not timezone aware:\n{}'
                    .format(field, dt, record)
                )
                raise ValueError(message)

    def to_records(self, dataframe):
        raise NotImplementedError('`to_records()` must be implemented.')

class ArbitrarySerializer(BaseSerializer):
    '''
    Arbitrary data at arbitrary non-overlapping intervals.
    Often used for montly billing data. Records must all have
    the "start" key and the "end" key. Overlaps are not allowed and
    gaps will be filled with NaN.

    For example::

        records = [
            {
                "start": datetime(2013, 12, 30, tzinfo=pytz.utc),
                "end": datetime(2014, 1, 28, tzinfo=pytz.utc),
                "value": 1180,
            },
            {
                "start": datetime(2014, 1, 28, tzinfo=pytz.utc),
                "end": datetime(2014, 2, 27, tzinfo=pytz.utc),
                "value": 1211,
                "estimated": True,
            },
            {
                "start": datetime(2014, 2, 27, tzinfo=pytz.utc),
                "end": datetime(2014, 3, 30, tzinfo=pytz.utc),
                "value": 985,
            },
            {
                "start": datetime(2014, 3, 30, tzinfo=pytz.utc),
                "end": datetime(2014, 4, 25, tzinfo=pytz.utc),
                "value": 848,
            },
            {
                "start": datetime(2014, 4, 25, tzinfo=pytz.utc),
                "end": datetime(2014, 5, 27, tzinfo=pytz.utc),
                "value": 533,
            },
            ...
        ]
    '''

    sort_key = "start"
    required_fields = ["start", "end", "value"]
    datetime_fields = ["start", "end"]

    def validate_record(self, record):
        super(ArbitrarySerializer, self)\
            .validate_record(record)

        if record["start"] >= record["end"]:
            message = 'Record "start" must be earlier than record "end":\n{}'\
                    '{} >= {}.'.format(record)
            raise ValueError(message)

    def validate_records(self, sorted_records):

        previous_end_datetime = None

        for record in sorted_records:

            self.validate_record(record)

            start = record["start"]
            end = record["end"]
            value = record["value"]
            estimated = record.get("estimated", False)

            if previous_end_datetime is None or start == previous_end_datetime:

                # normal record
                yield (start, value, estimated)
                previous_end_datetime = end

            elif start > previous_end_datetime:

                # blank record
                yield (previous_end_datetime, np.nan, False)

                # normal record
                yield (start, value, estimated)
                previous_end_datetime = end

            else: # start < previous_end_datetime
                message = 'Skipping overlapping record: '\
                        'start ({}) < previous end ({})'\
                        .format(start, previous_end_datetime)
                warnings.warn(message)

        # final record carries last datetime, but only if there was a record
        if previous_end_datetime is not None:
            yield (previous_end_datetime, np.nan, False)

    def to_records(self, df):
        records = []
        for s, e, v, est in zip(df.index, df.index[1:], df.value, df.estimated):
            records.append({
                "start": s.to_datetime(),
                "end": e.to_datetime(),
                "value": v,
                "estimated": bool(est),
            })
        return records

class ArbitraryStartSerializer(BaseSerializer):
    '''

    Arbitrary start data at arbitrary non-overlapping intervals.
    Records must all have the "start" key. The last data point
    will be ignored unless an end date is provided for it.
    This is useful for data dated to future energy use, e.g. billing for
    delivered fuels.

    For example::

        records = [
            {
                "start": datetime(2013, 12, 30, tzinfo=pytz.utc),
                "value": 1180,
            },
            {
                "start": datetime(2014, 1, 28, tzinfo=pytz.utc),
                "value": 1211,
            },
            {
                "start": datetime(2014, 2, 27, tzinfo=pytz.utc),
                "value": 985,
            },
            {
                "start": datetime(2014, 3, 30, tzinfo=pytz.utc),
                "value": 848,
            },
            {
                "start": datetime(2014, 4, 25, tzinfo=pytz.utc),
                "value": 533,
            },
            ...
        ]

    '''

    sort_key = "start"
    required_fields = ["start", "value"]
    datetime_fields = ["start"]

    def validate_records(self, sorted_records):

        n = len(sorted_records)
        for i, record in enumerate(sorted_records):

            self.validate_record(record)

            start = record["start"]
            value = record["value"]
            estimated = record.get("estimated", False)

            if i < n - 1: # all except last record
                yield (start, value, estimated)
            else: # last record
                end = record.get("end", None)
                if end is None:
                    # can't use the value of this record, no end date
                    yield (start, np.nan, False)
                else:
                    # provide an end date cap
                    if pd.notnull(value):
                        yield (start, value, estimated)
                        yield (end, np.nan, False)
                    else:
                        yield (start, np.nan, False)


    def to_records(self, df):
        records = []
        for i, row in df.iterrows():
            records.append({
                "start": i.to_datetime(),
                "value": row.value,
                "estimated": bool(row.estimated),
            })
        return records

class ArbitraryEndSerializer(BaseSerializer):
    '''

    Arbitrary end data at arbitrary non-overlapping intervals.
    Records must all have the "end" key. The first data point
    will be ignored unless a start date is provided for it.
    This is useful for data dated to past energy use, e.g. electricity
    or natural gas bills.

    Example::

        records = [
            {
                "end": datetime(2014, 1, 28, tzinfo=pytz.utc),
                "value": 1180,
            },
            {
                "end": datetime(2014, 2, 27, tzinfo=pytz.utc),
                "value": 1211,
            },
            {
                "end": datetime(2014, 3, 30, tzinfo=pytz.utc),
                "value": 985,
                "estimated": False,
            },
            {
                "end": datetime(2014, 4, 25, tzinfo=pytz.utc),
                "value": 848,
            },
            {
                "end": datetime(2014, 5, 27, tzinfo=pytz.utc),
                "value": 533,
            },
            ...
        ]

    '''

    sort_key = "end"
    required_fields = ["end", "value"]
    datetime_fields = ["end"]

    def validate_records(self, sorted_records):

        previous_end_datetime = None

        for record in sorted_records:

            self.validate_record(record)

            end = record["end"]
            value = record["value"]
            estimated = record.get("estimated", False)

            if previous_end_datetime is None:

                # first record, might have start
                start = record.get("start", None)

                if start is not None:
                    yield (start, value, estimated)
            else:
                yield (previous_end_datetime, value, estimated)

            previous_end_datetime = end

        if previous_end_datetime is not None:
            yield (previous_end_datetime, np.nan, estimated)


    def to_records(self, df):
        records = []

        if df.shape[0] > 0:
            records.append({
                "end": df.index[0].to_datetime(),
                "value": np.nan,
                "estimated": False,
            })

            for e, v, est in zip(df.index[1:], df.value, df.estimated):
                records.append({
                    "end": e.to_datetime(),
                    "value": v,
                    "estimated": bool(est),
                })

        return records


class EnergyTimeSeries(object):
    """ Container for energy time series data.

    Parameters
    ----------
    label : str
        An arbitrary identifier for this instance.
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
    ]

    def __init__(self, label, fuel, interpretation, data=None, unit=None, placeholder=False, serializer=None):

        self.label = label
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
                "EnergyTimeSeries({}, {})\n  PLACEHOLDER"
                .format(self.fuel, self.interpretation)
            )
        else:
            return (
                "EnergyTimeSeries({}, {})\n{}"
                .format(self.fuel, self.interpretation, self.data)
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

    # def filter_by_period(self, period):
    #     """ Return a new ConsumptionData instance within the period.
    #
    #     Parameters
    #     ----------
    #     period : eemeter.evaluation.Period
    #         Period within which to get ConsumptionData
    #
    #     Returns
    #     -------
    #     consumption_data : eemeter.consumption.ConsumptionData
    #         ConsumptionData instance holding data within the requested period.
    #     """
    #     filtered_data = None
    #     filtered_estimated = None
    #     if period.start is None and period.end is None:
    #         filtered_data = self.data.copy()
    #         filtered_estimated = self.estimated.copy()
    #     elif period.start is None and period.end is not None:
    #         filtered_data = self.data[:period.end].copy()
    #         filtered_estimated = self.estimated[:period.end].copy()
    #     elif period.start is not None and period.end is None:
    #         filtered_data = self.data[period.start:].copy()
    #         filtered_estimated = self.estimated[period.start:].copy()
    #     else:
    #         filtered_data = self.data[period.start:period.end].copy()
    #         filtered_estimated = self.estimated[period.start:period.end].copy()
    #     if self.freq is None and filtered_data.shape[0] > 0:
    #         filtered_data.iloc[-1] = np.nan
    #         filtered_estimated.iloc[-1] = np.nan
    #     filtered_consumption_data = ConsumptionData(
    #             records=None,
    #             fuel_type=self.fuel_type,
    #             unit_name=self.unit_name,
    #             data=filtered_data,
    #             estimated=filtered_estimated)
    #     return filtered_consumption_data

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
