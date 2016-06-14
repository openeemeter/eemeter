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

    def __init__(self, fuel, interpretation, data=None, unit=None,
                 placeholder=False, serializer=None):

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
            message = (
                'Unsupported interpretation: "{}".'
                .format(interpretation)
            )
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


class TraceSet(object):

    def __init__(self, traces, labels=None):

        if labels is None:
            labels = self._generate_default_labels(traces)

        self._validate_lengths(traces, labels)

        self.traces = {label: trace for label, trace in zip(labels, traces)}

    def _generate_default_labels(self, traces):
        return [str(i) for i, _ in enumerate(traces)]

    def _validate_lengths(self, traces, labels):
        # make sure zip doesn't miss any
        if len(traces) != len(labels):
            message = (
                'Should be the same number of labels as traces,'
                ' but got {} labels for {} traces.'
                .format(len(labels), len(traces))
            )
            raise ValueError(message)

    def get_traces(self):
        for label, trace in self.traces.items():
            yield label, trace
