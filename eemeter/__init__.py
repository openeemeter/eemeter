# Version
VERSION = (0, 1, 6)

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
