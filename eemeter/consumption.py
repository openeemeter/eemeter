from . import ureg, Q_
from collections import defaultdict

class DateRangeException(Exception): pass
class InvalidFuelTypeException(Exception): pass

class FuelType:
    """
    Class for representing fuel types
    """

    def __init__(self,name):
        self.name = name

    def __str__(self):
        return self.name

electricity = FuelType("electricity")
natural_gas = FuelType("natural_gas")
propane = FuelType("propane")

class Consumption:
    """
    Class for representing energy usage
    """

    def __init__(self,usage,unit_name,fuel_type,start,end,estimated=False):
        self.fuel_type = fuel_type
        self.start = start
        self.end = end
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
        """
        Return interally stored joules value in the given unit
        """
        return (self.joules * ureg.joules).to(ureg.parse_expression(unit)).magnitude

    def __lt__(self,other):
        return self.end < other.end

    def to(self,unit):
        """
        Return interally stored joules value in the given unit
        """
        return (self.joules * ureg.joules).to(ureg.parse_expression(unit)).magnitude

    @property
    def timedelta(self):
        """
        Return the timedelta between the start and end datetimes
        """
        return self.end - self.start

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

    def get(self,fuel_type):
        if isinstance(fuel_type,FuelType):
            return self._data.get(fuel_type.name)
        return self._data.get(fuel_type)

    def iteritems(self):
        for key,item in self._data.items():
            for consumption in item:
                yield consumption

    def fuel_types(self):
        return self._data.iteritems()
