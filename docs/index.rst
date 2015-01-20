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

This library uses the py.test framework. To develop locally, clone the repo,
and in a virtual environment execute the following commands:

    $ git clone https://github.com/philngo/ee-meter
    $ cd eemeter
    $ pip install numpy scipy pytest
    $ python setup.py develop
    $ py.test

You should ensure that you are using the virtualenv py.test executable with
`py.test --version`.

Some tests are slow and are skipped by default; to run these, use the `--runslow` flag:

    $ py.test --runslow

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Licence
-------

MIT
