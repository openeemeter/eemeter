from . import ureg, Q_

class DateRangeException(Exception): pass

class DatetimePeriod:
    """Class to represents a period of time with a start and an end. Used as the
    Base class for consumptions. When `DatetimePeriod` instances are compared,
    they are compared by end time (because bills come at the end of the month).
    """
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __lt__(self,other):
        """Comparison is based on end time.
        """
        return self.end < other.end

    def __eq__(self,other):
        return self.joules == other.joules and self.start == other.start and \
                self.end == other.end and self.fuel_type == other.fuel_type and \
                self.estimated == other.estimated
    @property
    def timedelta(self):
        """Property representing the timedelta between the start and end
        datetimes.
        """
        return self.end - self.start

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "DatetimePeriod({},{})".format(self.start,self.end)

class Consumption(DatetimePeriod):
    """Represents energy usage. Each instance has start and end datetimes, a
    particular unit (although it is stored internally as joules), a string
    identifying a `fuel_type` and whether or not it is estimated. (Sometimes
    estimated bills are treated differently).
    """

    def __init__(self,usage,unit_name,fuel_type,start,end,estimated=False):
        DatetimePeriod.__init__(self,start,end)
        self.fuel_type = fuel_type
        self.estimated = estimated

        quantity = usage * ureg.parse_expression(unit_name)

        self.joules = quantity.to(ureg.joules).magnitude

        if self.end < self.start:
            raise DateRangeException

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "Consumption ({} to {}): {} J".format(self.start.strftime("%Y-%m-%d"),
                                                     self.end.strftime("%Y-%m-%d"),
                                                     self.joules)

    def __getattr__(self,unit):
        return (self.joules * ureg.joules).to(ureg.parse_expression(unit)).magnitude

    def __iter__(self):
        raise TypeError("Consumption objects cannot act as iterators.")

    def to(self,unit):
        """Returns internally stored energy value in the given unit. The `unit`
        should be a representative string in abbreviation or otherwise, in
        singular or plural form (e.g. "kWh", "kWhs", "kilowatthour", or
        "kilowatthours"). Units must have the same dimensionality as
        the joule. Unit handling uses the `pint` package, which has additional
        documentation.
        """
        return (self.joules * ureg.joules).to(ureg.parse_expression(unit)).magnitude

    def average_daily_usage(self,unit):
        """Returns average daily usage over the period in the energy
        unit supplied in `unit`.
        """
        if self.timedelta.days == 0:
            return float('nan')
        return (self.joules * ureg.joules / self.timedelta.days).to(ureg.parse_expression(unit)).magnitude

class ConsumptionHistory:
    """Represents energy usage attributed to a single property or project.
    Separates usage by `fuel_type` and provides commonly-used filters and
    iterators over its data. Often used as inputs to meters.
    """
    def __init__(self,consumptions):
        self._data = {}
        for consumption in sorted(consumptions):
            if consumption.fuel_type in self._data:
                self._data[consumption.fuel_type].append(consumption)
            else:
                self._data[consumption.fuel_type] = [consumption]

    def __getattr__(self, attr):
        return self._data[attr]

    def __getitem__(self,item):
        return self._data[item]

    def __repr__(self):
        return "ConsumptionHistory({})".format(self._data)

    def __str__(self):
        return "ConsumptionHistory({})".format(self._data)

    def __nonzero__(self):
        return len(self._data.keys()) > 0

    def after(self,dt):
        """Returns a ConsumptionHistory object containing all consumptions
        which have start datetimes on or after the given datetime.
        """
        consumptions = []
        for item in self.iteritems():
            if item.start >= dt:
                consumptions.append(item)
        return ConsumptionHistory(consumptions)

    def before(self,dt):
        """Returns a ConsumptionHistory object containing all consumptions
        which have end datetimes on or before the given datetime.
        """
        consumptions = []
        for item in self.iteritems():
            if item.end <= dt:
                consumptions.append(item)
        return ConsumptionHistory(consumptions)

    def get(self,fuel_type):
        """Returns an array (not necessarily sorted) of Consumption instances
        given a particular fuel_type. Fuel type should be specified as a
        string such as `"electricity"` or `"natural_gas"`. Returns `None` if
        no matching `Consumption` instances are found.
        """
        return self._data.get(fuel_type,[])

    def iteritems(self):
        """Iterator that returns all internally stored consumption instances.
        """
        for fuel_type,consumptions in self._data.items():
            for consumption in consumptions:
                yield consumption

    def fuel_types(self):
        """Iterates over (fuel_type,consumptions) pairs in no particular order,
        in which fuel_type is the string reprenting a fuel_type (such as
        `"electricity"` or `"natural_gas"`) and consumptions is a list of all
        Consumption instances with that particular fuel type.
        """
        return self._data.items()
