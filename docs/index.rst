.. EEMeter documentation master file, created by
   sphinx-quickstart on Fri Oct 24 11:54:19 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Open Energy Efficiency Meter
============================

Description
-----------

This package makes it simple to build and maintain residential and commercial
energy efficieny monitoring systems that operate at scale.

It implements a number of building energy efficiency monitoring standards,
including the following standards:

- PRISM
- `ANSI/BPI-2400-S-2012 <http://www.bpi.org/Web%20Download/BPI%20Standards/BPI-2400-S-2012_Standard_Practice_for_Standardized_Qualification_of_Whole-House%20Energy%20Savings_9-28-12_sg.pdf>`_

Usage
-----

.. toctree::
   :maxdepth: 2

   tutorial

Installation
------------

Execute the following command to install the eemeter package and its dependencies::

   pip install git+git://github.com/philngo/ee-meter.git#egg=ee-meter

Testing
-------

Execute the following command to run tests::

   py.test

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
