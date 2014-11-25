Tutorial
========

To get started with the eemeter, install it with the following command::

    pip install eemeter

Formatting data
---------------

.. code-block:: python

    from eemeter.consumption import Consumption
    from eemeter.consumption import ConsumptionHistory

.. code-block:: python

    consumptions = [Consumption(),Consumption()]
    consumption_history = ConsumptionHistory(consumptions)
