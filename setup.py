from distutils.core import setup, find_packages

version = __import__('eemeter').get_version()

setup(name='EEMeter',
    version=version,
    description='Open energy efficiency meter',
    author='Matt Gee, Phil Ngo',
    packages=find_packages(),
    )
