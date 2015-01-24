from . import ureg, Q_
from collections import defaultdict

class DateRangeException(Exception): pass
class InvalidFuelTypeException(Exception): pass

class FuelType:
    """Simple representation of fuel types. Stores the name of the fuel type.
    """

    def __init__(self,name):
        self.name = name

    def __str__(self):
        return self.name

# weird psuedo-global singletons; should probably reconsider this implementation.
electricity = FuelType("electricity")
natural_gas = FuelType("natural_gas")
propane = FuelType("propane")

class DatetimePeriod:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __lt__(self,other):
        """Comparison is based on end time.
        """
        return self.end < other.end

    @property
    def timedelta(self):
        """Property representing the timedelta between the start and end
        datetimes.
        """
        return self.end - self.start

class Consumption(DatetimePeriod):
    """Represents energy usage. Each instance has start and end datetimes, a
    particular unit (although it is stored internally as joules), a fuel type,
    and whether or not it is estimated.
    """

    def __init__(self,usage,unit_name,fuel_type,start,end,estimated=False):
        DatetimePeriod.__init__(self,start,end)
        self.fuel_type = fuel_type
        self.estimated = estimated

        quantity = usage * ureg.parse_expression(unit_name)

        self.joules = quantity.to(ureg.joules).magnitude

        if self.end < self.start:
            raise DateRangeException

        if not isinstance(self.fuel_type,FuelType):
            raise InvalidFuelTypeException

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "Consumption ({} to {}): {} J".format(self.start.strftime("%Y-%m-%d"),
                                                     self.end.strftime("%Y-%m-%d"),
                                                     self.joules)

    def __getattr__(self,unit):
        return (self.joules * ureg.joules).to(ureg.parse_expression(unit)).magnitude

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
        return (self.joules * ureg.joules / self.timedelta.days).to(ureg.parse_expression(unit)).magnitude

class ConsumptionHistory:
    def __init__(self,consumptions):
        self._data = {}
        for consumption in consumptions:
            if consumption.fuel_type.name in self._data:
                self._data[consumption.fuel_type.name].append(consumption)
            else:
                self._data[consumption.fuel_type.name] = [consumption]

    def __getattr__(self, attr):
        return self._data[attr]

    def __getitem__(self,item):
        if isinstance(item,FuelType):
            return self._data[item.name]
        return self._data[item]

    def __repr__(self):
        return "ConsumptionHistory({})".format(self._data)

    def __str__(self):
        return "ConsumptionHistory({})".format(self._data)

    def __nonzero__(self):
        return len(self._data.keys()) > 0

    def after(self,dt):
        consumptions = []
        for item in self.iteritems():
            if item.start >= dt:
                consumptions.append(item)
        return ConsumptionHistory(consumptions)

    def before(self,dt):
        consumptions = []
        for item in self.iteritems():
            if item.end <= dt:
                consumptions.append(item)
        return ConsumptionHistory(consumptions)

    def get(self,fuel_type):
        """Returns an array (not necessarily sorted) of Consumption instances
        given a particular fuel_type. Fuel type may be specified as a string
        matching the name of a particular fuel type, or by a fuel type
        instance.
        """
        if isinstance(fuel_type,FuelType):
            return self._data.get(fuel_type.name)
        return self._data.get(fuel_type)

    def iteritems(self):
        """Iterator that returns all internally stored consumption instances in
        no particular order.
        """
        for fuel_type,consumptions in self._data.items():
            for consumption in consumptions:
                yield consumption

    def fuel_types(self):
        """Iterates over (fuel_type,consumptions) pairs, in which fuel_type is
        the string name of a fuel_type and consumptions is a list of all
        consumption with that particular fuel type.
        """
        return self._data.iteritems()
