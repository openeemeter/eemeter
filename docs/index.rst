.. eemeter documentation master file, created by
   sphinx-quickstart on Tue Feb 13 17:38:45 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. spelling::

   CDD

EEmeter: tools for calculating metered energy savings
=====================================================

.. image:: https://travis-ci.org/openeemeter/eemeter.svg?branch=master
    :target: https://travis-ci.org/openeemeter/eemeter

.. image:: https://img.shields.io/github/license/openeemeter/eemeter.svg
    :target: https://github.com/openeemeter/eemeter

.. image:: https://readthedocs.org/projects/eemeter/badge/?version=latest
    :target: http://eemeter.readthedocs.io/en/latest/?badge=latest

.. image:: https://img.shields.io/pypi/v/eemeter.svg
    :target: https://pypi.python.org/pypi/eemeter

.. image:: https://codecov.io/gh/openeemeter/eemeter/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/openeemeter/eemeter

---------------

**EEmeter** — open source implementations of standard methods for calculating
metered energy savings.

The eemeter contains the reference implementation of the CalTRACK methods for
computing metered energy usage differences at sites with building efficiency
interventions or at control sites without known interventions.

Installation
------------

EEmeter is a python package and can be installed with pip.

::

    $ pip install eemeter

Features
--------

- Candidate model selection
- Data sufficiency checking
- Reference implementation of standard methods

  - CalTRACK Daily Method
  - CalTRACK Monthly Method

- Flexible sources of temperature data. See :any:`EEweather <eeweather:index>`.
- Model serialization
- First-class warnings reporting
- Pandas DataFrame support
- Visualization tools

Command-line Usage
------------------

Once installed, ``eemeter`` can be run from the command-line. To see all available commands, run ``eemeter --help``.

Use CalTRACK methods on sample data::

    $ eemeter caltrack --sample=il-electricity-cdd-hdd-daily

Save output::

    $ eemeter caltrack --sample=il-electricity-cdd-only-billing_monthly --output-file=/path/to/output.json

Load custom data (see :any:`eemeter.meter_data_from_csv` and :any:`eemeter.temperature_data_from_csv` for formatting)::

    $ eemeter caltrack --meter-file=/path/to/meter/data.csv --temperature-file=/path/to/temperature/data.csv

Do not fit CDD-based candidate models (intended for gas data)::

    $ eemeter caltrack --sample=il-gas-hdd-only-billing_bimonthly --no-fit-cdd

Usage Guides
------------

.. toctree::
   :maxdepth: 2

   basics
   advanced
   api
