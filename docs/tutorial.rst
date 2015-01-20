Tutorial
========

Introduction
------------

The Open Energy Efficiency Meter is an engine for measurement and verification
of energy savings across and between energy efficiency programs for both
commercial and residential applications. At its core, this package is a
framework for specifying how efficiency metrics should be calculated from raw
consumption data. It is flexible enough to meet the needs of different program
implementers without sacrificing the ability to standardize on particular
models and methods, and adds the ability to generate realtime or near realtime
reports on realized energy savings.

The meter handles import of consumption data from various formats and
data-stores, including Green Button, HPXML, and SEED. It handles downloading
and local caching of realtime or near realtime weather data and normals,
including TMY3, GSOD, ISD, wunderground.com, degreedays.net.

Although the package comes with pre-written implementations of several
efficiency metering standards, custom meters can be written by assembling
existing meter components or by writing new components.

Installation
------------

To get started with the eemeter, use pip::

    $ pip install git+git://github.com/philngo/ee-meter.git#egg=ee-meter

or download it from github and install it using the setup.py::

    $ git clone git://github.com/philngo/ee-meter.git
    $ cd ee-meter/
    $ python setup.py install

Using an existing meter
-----------------------

TODO

Running a meter
---------------

TODO

Creating a custom meter
-----------------------

TODO

