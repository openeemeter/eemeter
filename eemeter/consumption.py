from . import ureg, Q_

from datetime import datetime
import pandas as pd
import numpy as np
from warnings import warn
import pytz

from eemeter.evaluation import Period

class ConsumptionData(object):
    """ Container for consumption data initialized from records.
    Warns about overlapping data, and allows for a few different ways
    of specifying consumption data.

    Parameters
    ----------

    records : list of dicts
        List of records (sorted or unsorted). Each record is a dict
        with the keys "start", "end", "value", "pulse", or "estimated".
        - "start", "end", "start" and "end", or "pulse"
          (datetime.datetime) should define the time period.
          See the argument 'record_type' of this object for more
          detail on time period format.
        - "value" (float) should be the amount of consumption during
          the given time period.
        - "estimated" (boolean, optional) should indicate whether or
          not the bill (if relevant) was estimated.
    fuel_type : str, {"electricity", "natural_gas"}
        The fuel type of the consumption data.
    unit_name : str, {"kWh", "therm"}
        The name of the unit in which the consumption data is given.
    record_type : str, {'interval', 'arbitrary', 'pulse'}
        - 'interval': data at regular time intervals. Records must
          all have the "start" key or must all have the "end" key.
        - 'arbitrary': data at arbitrary non-overlapping intervals.
          Often used for montly billing data. Records must all have
          the "start" key and the "end" key.
        - 'arbitrary_start': data at arbitrary non-overlapping intervals.
          Records must all have the "start" key. The last data point
          will be ignored unless an end date is provided for it.
          This is useful for data dated
          to future energy use, e.g. billing for delivered fuels.
        - 'arbitrary_end': data at arbitrary non-overlapping intervals.
          Records must all have the "end" key. The first data point
          will be ignored unless a start date is provided for it.
          This is useful for data dated to past energy use,
          e.g. electricity or natural gas bills.
        - 'pulse': data at regular consumption intervals. Records
          must all have the "pulse" key.
        - 'billing': Alias for 'arbitrary'.
        - 'billing_start': Alias for 'arbitrary_start'.
        - 'billing_end': Alias for 'arbitrary_end'.
    freq : str
        String representing the frequency of intervals; should be one
        of the following pandas offset alias options.
        Used for record_type="interval".

        - 'D': calendar day frequency
        - 'H': hourly frequency
        - 'T': minutely frequency (not to be confused with 'M', which
          means "month end frequency"
        - 'S': secondly frequency

        You may also add an integer string in front of the
        frequency marker (e.g. '15T' ==> 15 minutes)
    pulse_value : float
        The value of a single pulse. Used for record_type="pulse".
    name : str, default None
        An identifier for this instance of Consumption Data.
    data : str, default None
        Pre-parsed consumption data. Please also set records=None.
    """

    def __init__(self, records, fuel_type, unit_name,
            record_type="interval", freq=None, pulse_value=None, name=None,
            data=None, estimated=None):

        # verify and save unit name
        if unit_name not in ["kWh", "therm"]:
            message = 'Unsupported unit name: "{}".'.format(unit_name)
            raise ValueError(message)
        else:
            self.unit_name = unit_name

        # verify and save fuel type
        if fuel_type not in ["electricity", "natural_gas"]:
            message = 'Unsupported fuel type: "{}".'.format(fuel_type)
            raise ValueError(message)
        else:
            self.fuel_type = fuel_type

        # set misc attributes
        self.freq = freq
        self.pulse_value = pulse_value
        self.name = name

        # import data directly (skipping record parsing) if available.
        if data is not None:
            if not records is None:
                message = "Please provide either data or records, but not both."
                raise ValueError(message)
            if estimated is None:
                message = "Please provide the the `estimated` attribute," \
                        " which contains boolean values indicating whether" \
                        " or not the data is estimated. Should have the same" \
                        " index as `data`"
                raise ValueError(message)
            self.data = data
            self.estimated = estimated

            # set frequency, if supplied.
            if freq is None:
                self.freq_timedelta = None
            elif freq[-1] not in ["D","H","T","S"]:
                # Improper configuration
                message = 'Invalid frequency specification: "{}".'.format(freq)
                raise ValueError(message)
            else:
                try:
                    dummy_start_date = datetime(1970,1,1,tzinfo=pytz.utc)
                    dummy_date_range = pd.date_range(dummy_start_date,
                            periods=2, freq=freq)
                    freq_timedelta = dummy_date_range[1] - dummy_date_range[0]
                except ValueError:
                    message = 'Invalid frequency specification: "{}".'\
                            .format(freq)
                    raise ValueError(message)
                self.freq_timedelta = freq_timedelta
            return

        # import records
        if "interval" == record_type:
            if freq is None or freq[-1] not in ["D","H","T","S"]:
                # Improper configuration
                message = 'Invalid frequency specification: "{}".'.format(freq)
                raise ValueError(message)
            else:
                try:
                    dummy_start_date = datetime(1970,1,1,tzinfo=pytz.utc)
                    dummy_date_range = pd.date_range(dummy_start_date,
                            periods=2, freq=freq)
                    freq_timedelta = dummy_date_range[1] - dummy_date_range[0]
                except ValueError:
                    message = 'Invalid frequency specification: "{}".'\
                            .format(freq)
                    raise ValueError(message)
            self.freq_timedelta = freq_timedelta
            self.data, self.estimated = self._import_interval(records)
        elif record_type in ["arbitrary", "billing"]:
            self.freq_timedelta = None
            self.data, self.estimated = self._import_arbitrary(records)
        elif record_type in ["arbitrary_start", "billing_start"]:
            self.freq_timedelta = None
            self.data, self.estimated = self._import_arbitrary_start(records)
        elif record_type in ["arbitrary_end", "billing_end"]:
            self.freq_timedelta = None
            self.data, self.estimated = self._import_arbitrary_end(records)
        elif "pulse" == record_type:
            if pulse_value is None or pulse_value <= 0:
                # incorrectly configured
                message = 'Expected pulse_value to be a positive float, '\
                        'but got {} instead.'.format(pulse_value)
                raise ValueError(message)
            self.freq_timedelta = None
            self.data, self.estimated = self._import_pulse(records)
        else:
            message('Invalid record_type: "{}".'.format(record_type))
            raise ValueError

    def _import_interval(self, records):
        if records == []:
            return pd.Series([]), pd.Series([])
        if records[0].get("start") is not None:
            key = "start"
        elif records[0].get("end") is not None:
            key = "end"
        else:
            message = 'Records must all have a "start" key or must'\
                      ' all have a "end" key.'
            raise ValueError(message)
        try:
            sorted_records = sorted(records, key=lambda x: x[key])
        except KeyError:
            message = 'Records must all have a "start" key or must'\
                      ' all have a "end" key.'
            raise ValueError(message)
        start_limit = sorted_records[0].get(key)
        end_limit = sorted_records[-1].get(key)
        dt_index = pd.date_range(start=start_limit, end=end_limit,
                freq=self.freq)

        # shift index back if the keys given are end dates.
        if key == "end":
            dt_index -= self.freq_timedelta

        data = pd.Series(np.tile(np.nan, dt_index.shape),
                index=dt_index)
        estimated = pd.Series(np.tile(False, dt_index.shape),
                index=dt_index)
        for record in sorted_records:
            dt = record.get(key)
            value = record.get("value")
            est = record.get("estimated")
            if dt is None:
                message = 'Records must all have a "start" key or'\
                          ' must all have a "end" key.'
                raise ValueError(message)
            if key == "end":
                dt -= self.freq_timedelta
            try:
                current_value = data[dt]
            except KeyError:
                current_value = None
            if current_value is None:
                message = "Ignoring misaligned data point:"\
                    " (data[{}] = {})".format(dt,value)
                warn(message)
            elif pd.isnull(current_value):
                data[dt] = value
                if est:
                    estimated[dt] = True
            else:
                message = "Ignoring overlapping data point:"\
                    " (data[{}] = {})".format(dt,value)
                warn(message)
        return data, estimated

    def _import_arbitrary(self, records):
        if records == []:
            return pd.Series([]), pd.Series([])
        try:
            sorted_records = sorted(records, key=lambda x: x["start"])
        except KeyError:
            message = 'Records must all have a "start" key and an'\
                    ' "end" key.'
            raise ValueError(message)
        start_datetimes = []
        values = []
        estimateds = []
        previous_end_datetime = None
        for record in sorted_records:
            start = record.get("start")
            end = record.get("end")
            value = record.get("value")
            estimated = record.get("estimated")
            if start is None or end is None:
                message = 'Records must all have a "start" key and an'\
                        ' "end" key.'
                raise ValueError(message)
            else:
                if start >= end:
                    message = 'Record start must be earlier than end:'\
                            '{} >= {}.'.format(start,end)
                    raise ValueError(message)
            if previous_end_datetime is None or\
                    start == previous_end_datetime:
                start_datetimes.append(start)
                values.append(value)
                previous_end_datetime = end
                estimateds.append(bool(estimated))
            elif start < previous_end_datetime:
                message = 'Skipping overlapping record: '\
                        'start ({}) < previous end ({})'\
                        .format(start,previous_end_datetime)
                warn(message)
            else: # start > previous_end_datetime:
                start_datetimes.append(previous_end_datetime)
                values.append(np.nan)
                estimateds.append(False)
                start_datetimes.append(start)
                values.append(value)
                previous_end_datetime = end
                estimateds.append(bool(estimated))
        # append a NaN to represent the end date of the last one.
        start_datetimes.append(previous_end_datetime)
        values.append(np.nan)
        estimateds.append(False)
        dt_index = pd.DatetimeIndex(start_datetimes)
        data = pd.Series(values, index=dt_index)
        estimated = pd.Series(estimateds, index=dt_index)
        return data, estimated

    def _import_arbitrary_start(self, records):
        if records == []:
            return pd.Series([]), pd.Series([])
        try:
            sorted_records = sorted(records, key=lambda x: x["start"])
        except KeyError:
            message = 'Records must all have a "start" key.'
            raise ValueError(message)
        start_datetimes = []
        values = []
        estimateds = []
        for record in sorted_records:
            start = record["start"]
            value = record.get("value")
            estimated = record.get("estimated")
            start_datetimes.append(start)
            values.append(value)
            estimateds.append(bool(estimated))
        # append an element if the last record has an end date.
        last_end = sorted_records[-1].get("end")
        if last_end is None:
            values[-1] = np.nan
        else:
            start_datetimes.append(last_end)
            values.append(np.nan)
            estimateds.append(False)
        dt_index = pd.DatetimeIndex(start_datetimes)
        data = pd.Series(values, index=dt_index)
        estimated = pd.Series(estimateds, index=dt_index)
        return data, estimated

    def _import_arbitrary_end(self, records):
        if records == []:
            return pd.Series([]), pd.Series([])
        try:
            sorted_records = sorted(records, key=lambda x: x["end"])
        except KeyError:
            message = 'Records must all have a "end" key.'
            raise ValueError(message)
        end_datetimes = [r["end"] for r in sorted_records]
        values = [r.get("value") for r in sorted_records[1:]]
        estimateds = [bool(r.get("estimated")) for r in sorted_records[1:]]

        # insert start value to if first record has a start date
        first_start = sorted_records[0].get("start")
        if first_start is not None:
            end_datetimes.insert(0,first_start)
            values.insert(0,sorted_records[0].get("value"))
            estimateds.insert(0,bool(sorted_records[0].get("estimated")))
        values.append(np.nan)
        estimateds.append(False)
        dt_index = pd.DatetimeIndex(end_datetimes)
        data = pd.Series(values, index=dt_index)
        estimated = pd.Series(estimateds, index=dt_index)
        return data, estimated

    def _import_pulse(self, records):
        if records == []:
            return pd.Series([]), pd.Series([])
        try:
            sorted_records = sorted(records, key=lambda x: x["pulse"])
        except KeyError:
            message = 'Records must all have a "pulse" key.'
            raise ValueError(message)
        pulses = [r.get("pulse") for r in sorted_records]
        estimateds = [bool(r.get("estimated")) for r in sorted_records]
        values = np.tile(float(self.pulse_value),(len(pulses),))
        values[0] = np.nan # the first pulse is treated as the first start
        dt_index = pd.DatetimeIndex(pulses)
        data = pd.Series(values,index=dt_index)
        estimated = pd.Series(estimateds, index=dt_index)
        return data, estimated

    def to(self,unit_name):
        old_quantities = Q_(self.data.values, ureg[self.unit_name])
        new_quantities = old_quantities.to(unit_name)
        return new_quantities.magnitude

    def periods(self):
        if self.freq_timedelta is None:
            # ignore last period which is NaN and acting as an end point
            periods = [Period(start, end) for start, end in
                       zip(self.data.index,self.data.index[1:])]
            return periods
        else:
            periods = [Period(dt, dt + self.freq_timedelta)
                       for dt in self.data.index]
            return periods

    def average_daily_consumptions(self):
        if self.freq_timedelta is None:
            # ignore last period which is NaN and acting as an end point
            avgs, n_days = [], []
            for v, ns in zip(self.data,np.diff(self.data.index)):
                # nanoseconds to days
                days = ns.astype('d')/8.64e13
                avgs.append(v/days)
                n_days.append(days)
            return np.array(avgs), np.array(n_days)
        else:
            days = self.freq_timedelta.days + self.freq_timedelta.seconds/8.64e4
            avgs, n_days = [], []
            for v in self.data:
                avgs.append(v/days)
                n_days.append(days)
            return np.array(avgs), np.array(n_days)

    def total_period(self):
        if self.data.shape[0] < 1:
            return None
        start_date = self.data.index[0]
        end_date = self.data.index[-1]
        if self.freq_timedelta is not None:
            end_date += self.freq_timedelta
        return Period(start_date, end_date)

    def total_days(self):
        tdelta = self.total_period().timedelta
        return tdelta.days + tdelta.seconds/8.64e4

    def records(self, record_type="arbitrary"):
        records = []
        if record_type == "interval":
            for i,v in self.data.iteritems():
                records.append({"start": i, "value": v})
        elif record_type in ["arbitrary", "billing"]:
            for v,s,e in zip(self.data,self.data.index,self.data.index[1:]):
                records.append({"start": s,"end": e,"value": v})
        elif record_type in ["arbitrary_start", "billing_start"]:
            for i,v in self.data.iteritems():
                records.append({"start": i,"value": v})
        elif record_type in ["arbitrary_end", "billing_end"]:
            records.append({"end": self.data.index[0], "value": np.nan})
            for v,e in zip(self.data,self.data.index[1:]):
                records.append({"end": e,"value": v})
        elif record_type == "pulse":
            for i in self.data.index:
                records.append({"pulse": i})
            shape = (self.data.values.shape[0] - 1,)
            if len(records) > 1 and not all(self.data.values[1:] == \
                    np.tile(self.data.values[1], shape)):
                message = 'record_type="pulse" implies that all values' \
                        ' should be the same, but they are not: {}'\
                        .format(self.data.values)
                warn(message)
        else:
            message = "Unsupported record_type: {}".format(record_type)
            raise ValueError(message)
        return records

    def filter_by_period(self, period):
        filtered_data = None
        filtered_estimated = None
        if period.start is None and period.end is None:
            filtered_data = self.data.copy()
            filtered_estimated = self.estimated.copy()
        elif period.start is None and period.end is not None:
            filtered_data = self.data[:period.end].copy()
            filtered_estimated = self.estimated[:period.end].copy()
        elif period.start is not None and period.end is None:
            filtered_data = self.data[period.start:].copy()
            filtered_estimated = self.estimated[period.start:].copy()
        else:
            filtered_data = self.data[period.start:period.end].copy()
            filtered_estimated = self.data[period.start:period.end].copy()
        if self.freq is None and filtered_data.shape[0] > 0:
            filtered_data.iloc[-1] = np.nan
            filtered_estimated.iloc[-1] = np.nan
        filtered_consumption_data = ConsumptionData(
                records=None,
                fuel_type=self.fuel_type,
                unit_name=self.unit_name,
                data=filtered_data,
                estimated=filtered_estimated)
        return filtered_consumption_data
