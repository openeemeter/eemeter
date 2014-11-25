Consumption data structures
===========================

Consumption is tracked by start and end date, units, usage, and fuel type.

Some fuel types have been built in:

.. autoclass:: eemeter.consumption.FuelType
   :members:

- electricity
- natural_gas

Instances of the Consumption class generally contain a single meter reading.

.. autoclass:: eemeter.consumption.Consumption
   :members:

The ConsumptionHistory class is created from an array of consumptions of
mixed or single fuel type.

.. autoclass:: eemeter.consumption.ConsumptionHistory
   :members:

Meters and Metrics
==================

Consumption is metered using meter classes which are containers for flags and
metrics. To create a new meter, use something similar to the following
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

.. autoclass:: eemeter.meter.MetricBase
   :members:
