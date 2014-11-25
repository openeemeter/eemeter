Consumption data structures
===========================

Consumption is tracked by start and end date, units, usage, and fuel type.

Some fuel types have been built in:

.. automodule:: eemeter.consumption
   :members:

Instances of the Consumption class generally contain a single meter reading.

.. autoclass:: eemeter.consumption.Consumption
   :members:

   .. automethod:: __init__

The ConsumptionHistory class is created from an array of consumptions of
mixed or single fuel type.

.. autoclass:: eemeter.consumption.ConsumptionHistory
   :members:

   .. automethod:: __init__

Meters and Metrics
==================

Consumption is metered using meter classes which are containers for flags and
metrics.

.. automodule:: eemeter.meter
   :members:
