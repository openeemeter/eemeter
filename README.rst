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

**EEmeter** — an open source python library for creating standardized models for 
predicting energy usage. These models are often used to calculate energy savings 
post demand side intervention (such as energy efficiency projects or demand 
response events).

Background - why use the EEMeter library
----------------------------------------

OpenEEmeter, as implemented in the eemeter package and sibling 
`eeweather package <http://eeweather.openee.io>`_ builds upon the foundation of the 
`CalTRACK Methods <https://caltrack.org/>`_ to provide free, open-source modeling tools
to anyone seeking to model energy building usage. Eemeter models have been developed to
meet or exceed the predictive capability of the CalTRACK models. These models adhere to 
a statistical approach, as opposed to an engineering approach, so that these models 
can be efficiently run on millions of meters at a time, while still providing 
accurate predictions. 

Using default settings in eemeter will provide accurate and stable model predictions 
suitable for savings measurements from demand side interventions. Settings can be 
modified for research and development purposes, although the outputs of such models 
may no longer be an officially recognized measurement as these models have been
verified by the OpenEEmeter Working Group.

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

- Models:

  - Energy Efficiency Daily Model
  - Energy Efficiency Billing (Monthly) Model
  - Energy Efficiency Hourly Model
  - Demand Response Hourly Model

- Flexible sources of temperature data. See `EEweather <https://eeweather.openee.io>`_.
- Data sufficiency checking
- Model serialization
- First-class warnings reporting
- Pandas dataframe support
- Visualization tools

Documentation
-------------

Documenation for this library can be found `here <https://openeemeter.github.io/eemeter/>`_.
Additionally, within the repository, the scripts directory contains Jupyter Notebooks, which
function as interactive examples.


Roadmap for 2024 development
----------------------------

The OpenEEmeter project growth goals for the year fall into two categories:

1. Community goals - we want help our community thrive and continue to grow.
2. Technical goals - we want to keep building the library in new ways that make it
   as easy as possible to use.

Community goals
---------------

1. Develop project documentation and tutorials

A number of users have expressed how hard it is to get started when tutorials are
out of date. We will dedicate time and energy this year to help create high quality
tutorials that build upon the API documentation and existing tutorials.

2. Make it easier to contribute

As our user base grows, the need and desire for users to contribute back to the library
also grows, and we want to make this as seamless as possible. This means writing and
maintaining contribution guides, and creating checklists to guide users through the
process.


Technical goals
~~~~~~~~~~~~~~~

1. Implement new OpenEEmeter models

The OpenEEmeter Working Group continues to improve the underlying models in 
OpenEEmeter. We seek to continue to implement these models in a safe, tested manner
so that these models may continue to be used within engineering pipelines effectively.

2. Weather normal and unusual scenarios

The EEweather package, which supports the OpenEEmeter, comes packaged with publicly
available weather normal scenarios, but one feature that could help make that easier
would be to package methods for creating custom weather year scenarios.

3. Greater weather coverage

The weather station coverage in the EEweather package includes full coverage of US and
Australia, but with some technical work, it could be expanded to include greater, or
even worldwide coverage.

License
-------

This project is licensed under [Apache 2.0](LICENSE).

Other resources
---------------

- `CONTRIBUTING <CONTRIBUTING.md>`_: how to contribute to the project.
- `MAINTAINERS <MAINTAINERS.md>`_: an ordered list of project maintainers.
- `CHARTER <CHARTER.md>`_: open source project charter.
- `CODE_OF_CONDUCT <CODE_OF_CONDUCT.md>`_: Code of conduct for contributors.
