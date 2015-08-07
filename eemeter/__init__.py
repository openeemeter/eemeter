# Version
VERSION = (0, 1, 8)

def get_version():
    return '{}.{}.{}'.format(VERSION[0], VERSION[1], VERSION[2])

# Units
try:
    from pint import UnitRegistry
    ureg = UnitRegistry()
    Q_ = ureg.Quantity
except ImportError:
    import warnings
    warnings.warn("Skipping pint to get version for setuptools.")
    ureg = None
    Q_ = None

from eemeter.evaluation import Period
from eemeter.consumption import ConsumptionData
from eemeter.project import Project
from eemeter.config.yaml_parser import load
from eemeter.meter.base import DataContainer
from eemeter.meter.base import DataCollection

