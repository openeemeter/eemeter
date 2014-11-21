# Units
from pint import UnitRegistry
ureg = UnitRegistry()
Q_ = ureg.Quantity

# Version
VERSION = (0, 0, 1)

def get_version():
    return '{}.{}.{}'.format(VERSION[0], VERSION[1], VERSION[2])
