from setuptools import setup, find_packages

version = __import__('eemeter').get_version()

setup(name='OpenEEMeter',
    version=version,
    description='Open Energy Efficiency Meter',
    author='Matt Gee, Phil Ngo, Eric Potash',
    packages=find_packages(),
    install_requires=['pint >= 0.6',
                      'pyyaml >= 3.11',
                      'numpy >= 1.9',
                      'scipy >= 0.15',
                      'requests >= 2.5'],
    package_data={'': ['*.json']},
)
