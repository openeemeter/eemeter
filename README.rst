EEmeter: tools for calculating metered energy savings
=====================================================

.. image:: https://travis-ci.org/openeemeter/eemeter.svg?branch=master
  :target: https://travis-ci.org/openeemeter/eemeter
  :alt: Build Status

.. image:: https://img.shields.io/github/license/openeemeter/eemeter.svg
  :target: https://github.com/openeemeter/eemeter
  :alt: License

.. image:: https://readthedocs.org/projects/eemeter/badge/?version=master
  :target: https://eemeter.readthedocs.io/?badge=master
  :alt: Documentation Status

.. image:: https://img.shields.io/pypi/v/eemeter.svg
  :target: https://pypi.python.org/pypi/eemeter
  :alt: PyPI Version

.. image:: https://codecov.io/gh/openeemeter/eemeter/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/openeemeter/eemeter
  :alt: Code Coverage Status

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
  :target: https://github.com/ambv/black
  :alt: Code Style

---------------

**EEmeter** — an open source toolkit for implementing and developing standard
methods for calculating normalized metered energy consumption (NMEC) and
avoided energy use.

Background - why use the EEMeter library
----------------------------------------

At time of writing (Sept 2018), the OpenEEmeter, as implemented in the eemeter
package and sister `eeweather package <http://eeweather.openee.io>`_, contains the
most complete open source implementation of the
`CalTRACK Methods <https://caltrack.org/>`_, which
specify a family of ways to calculate and aggregate estimates avoided energy
use at a single meter particularly suitable for use in pay-for-performance
(P4P) programs.

The eemeter package contains a toolkit written in the python langage which may
help in implementing a CalTRACK compliant analysis.

It contains a modular set of of functions, parameters, and classes which can be
configured to run the CalTRACK methods and close variants.

.. note::

    Please keep in mind that use of the OpenEEmeter is neither necessary nor
    sufficient for compliance with the CalTRACK method specification. For example,
    while the CalTRACK methods set specific hard limits for the purpose of
    standardization and consistency, the EEmeter library can be configured to edit
    or entirely ignore those limits. This is becuase the emeter package is used not
    only for compliance with, but also for *development of* the CalTRACK methods.

    Please also keep in mind that the EEmeter assumes that certain data cleaning
    tasks specified in the CalTRACK methods have occurred prior to usage with the
    eemeter. The package proactively exposes warnings to point out issues of this
    nature where possible.

Installation
------------

EEmeter is a python package and can be installed with pip.

::

    $ pip install eemeter

Features
--------

- Reference implementation of standard methods

  - CalTRACK Daily Method
  - CalTRACK Monthly Billing Method
  - CalTRACK Hourly Method

- Flexible sources of temperature data. See `EEweather <https://eeweather.openee.io>`_.
- Candidate model selection
- Data sufficiency checking
- Model serialization
- First-class warnings reporting
- Pandas dataframe support
- Visualization tools

License
-------

This project is licensed under [Apache 2.0](LICENSE).

Other resources
---------------

- `CONTRIBUTING <CONTRIBUTING.md>`_: how to contribute to the project.
- `MAINTAINERS <MAINTAINERS.md>`_: an ordered list of project maintainers.
- `CHARTER <CHARTER.md>`_: open source project charter.
- `CODE_OF_CONDUCT <CODE_OF_CONDUCT.md>`_: Code of conduct for contributors.
