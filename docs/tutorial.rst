Installation
========

To get started with the eemeter, install it with the following command::

    pip install eemeter

Meters, Metrics, and Flags
==========================

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

    meter = MyMeter()
    consumption_history = ConsumptionHistory([Consumption(1000.0,"kWh",electricity,datetime(2014,1,1),datetime(2014,2,1))])
    result = meter.run(consumption_history)

    # >>> print result.elec_data_present
    # True
    # >>> print result.elec_avg_usage
    # 1000.0000000000001

To create new metrics, subclass the `MetricBase` class and write the
`evaluate_fuel_type(consumptions)` method to suit your needs.

Consumption and Consumption History
===================================

Meters are data-source agnostic, but require input data to be in a specific
format. The `Consumption` and `ConsumptionHistory` classes take care of this
formatting. Using a meter, as shown in the example above, requires that the
consumption history be parsed and supplied from various data formats.

A set of importing tools and examples simplify this process, as consumption
data may originate from a variety of very different sources, including
wirelessly connected smart meters, specially formatted databases,
Home Performance XML, or CSV downloads from energy providers.
