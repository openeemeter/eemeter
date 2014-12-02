Consumption data structures
===========================

Overview
--------

Meters are data-source agnostic, but many built in metrics require input data
to be in the `ConsumptionHistory` format. A set of importing tools and examples
simplify the process of creating these python data containers, as consumption
data may originate from a variety of very different sources, including
wirelessly connected smart meters, specially formatted databases, Home
Performance XML, or CSV spreadsheets downloads from energy providers.

API reference
-------------

Consumption is tracked by start and end date, units, usage, and fuel type.

Some fuel types have been built in:

.. autoclass:: eemeter.consumption.FuelType
   :members:

- electricity
- natural_gas
- propane

Instances of the Consumption class generally contain a single meter reading.

.. autoclass:: eemeter.consumption.Consumption
   :members:

The ConsumptionHistory class is created from an array of consumptions of
mixed or single fuel type.

.. autoclass:: eemeter.consumption.ConsumptionHistory
   :members:

TODO
----

Add importers
Add database connections (SEED)
Add numpy array support
Add pandas time series support
