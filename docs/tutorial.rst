Tutorial
========

Installation
------------

To get started with the eemeter, install it with the following command::

    pip install eemeter

Creating new Meters
-------------------

Meters form the core of the eemeter functionality. Meters are effectively
groupings of particular metrics, procedures, and flags into executable and
parallelizable packages, abstracting the functional specification of energy
efficiency metrics away from the handling of different data sources, unit
systems, and fuel types.

The scope of metrics which can be implemented within the Meter framework is
nearly unlimited, ranging from simple flags to implementations of entire
procedures, such as the BPI-2400 energy efficiency specification.

The package comes out of the box with fully functional meters, but meters can
(and often will) be specified for particular needs.

To create a new meter, use something similar to the following
pattern, by subclassing the `eemeter.meter.Meter` class:

.. code-block:: python

    from eemeter.consumption import natural_gas
    from eemeter.consumption import electricity
    from eemeter.consumption import Consumption
    from eemeter.consumption import ConsumptionHistory

    from eemeter.meter import Meter
    from eemeter.meter import RawAverageUsageMetric
    from eemeter.meter import FuelTypePresenceFlag

    from datetime import datetime

    class MyMeter(Meter):
        elec_avg_usage = RawAverageUsageMetric("kWh",fuel_type=electricity)
        gas_avg_usage = RawAverageUsageMetric("therms",fuel_type=natural_gas)
        elec_data_present = FuelTypePresenceFlag(electricity)
        gas_data_present = FuelTypePresenceFlag(natural_gas)
        average_temperature = AverageTemperatureMetric(electricity)

    meter = MyMeter()
    consumption_history = ConsumptionHistory([Consumption(1000.0,"kWh",electricity,datetime(2014,1,1),datetime(2014,2,1))])
    gsod_weather_getter = GSODWeatherGetter('722874-93134',start_year=2014,end_year=2014)
    result = meter.run(consumption_history=consumption_history,weather_getter=gsod_weather_getter)

    # >>> print result.elec_data_present
    # True
    # >>> print result.elec_avg_usage
    # 1000.0000000000001

To create new metrics, subclass the `MetricBase` class and write the
`evaluate_fuel_type(consumptions)` method to suit your needs.

Creating new metrics and flags
------------------------------

To create new metrics or flags, inherit from the `MetricBase` or `FlagBase`
classes, and override either the `evaluate` or `evaluate_fuel_type` methods (the
latter of which simply takes care of some boilerplate for methods which run the
same evaluation for multiple (or specific) fuel types.

The meter class will automatically collect the named arguments and the `run`
method will take care of dispatching these arguments. Take care to avoid
argument name overlaps.
