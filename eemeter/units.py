from eemeter.conversions import kwh_to_BTU
from eemeter.conversions import BTU_to_kwh
from eemeter.conversions import therm_to_BTU
from eemeter.conversions import BTU_to_therm

class EnergyUnit:
    """
    Class for representing energy units and conversions
    """

    def __init__(self,full_name,abbreviation,toBTU,fromBTU):
        self.full_name = full_name
        self.abbreviation = abbreviation
        self._toBTU = toBTU
        self._fromBTU = fromBTU

    def __str__(self):
        return self.abbreviation

    def toBTU(self,value):
        """
        Return value in BTUs
        """
        return self._toBTU(value)

    def fromBTU(self,value):
        """
        Return Btu value in this unit
        """
        return self._fromBTU(value)


kWh = EnergyUnit("KilowattHour", "kWh", kwh_to_BTU, BTU_to_kwh)
therm = EnergyUnit("Therm", "therm", therm_to_BTU, BTU_to_therm)
BTU = EnergyUnit("BritishThermalUnit", "BTU", lambda x: x,lambda x: x)
