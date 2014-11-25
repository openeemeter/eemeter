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

