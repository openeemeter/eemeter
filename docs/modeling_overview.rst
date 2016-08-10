.. _modeling-overview:

Modeling Overview
-----------------

Model a single segment of an energy trace at a time

Types of models
^^^^^^^^^^^^^^^


Model error
^^^^^^^^^^^

Seasonal effects model
^^^^^^^^^^^^^^^^^^^^^^

When we measure savings using daily or hourly usage increments, additional
controls are required. For example, energy use on weekends is typically
systematically different than on weekdays and must be treated differently.
Likewise, overnight energy use looks very different than daytime energy use
even when weather conditions are the same. For these types of systematic
differences, we introduce a number of new methods. The most straightforward
is a "fixed-effects" method that takes into account the day of the week. This
and other modifications to the core method are designed to produce a more
accurate model for establishing a relationship between weather conditions and
energy usage, thus making our savings estimates more reliable.


.. _data-sufficiency:

Data sufficiency
^^^^^^^^^^^^^^^^

