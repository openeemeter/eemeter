.. EEMeter documentation master file, created by
   sphinx-quickstart on Fri Oct 24 11:54:19 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Open Energy Efficiency Meter
============================

.. warning::

   The `eemeter` package is under rapid development; we are working quickly
   toward a stable release. In the mean time, please proceed to use the package,
   but as you do so, recognize that the API is in flux and the docs might not
   be up-to-date. Feel free to contribute changes or open issues on
   `github <https://github.com/impactlab/eemeter>`_ to report bugs, request
   features, or make suggestions.

Description
-----------

This package makes it simple to build and maintain residential and commercial
energy efficieny monitoring systems that operate at scale.

Usage
-----

.. toctree::
   :maxdepth: 2

   tutorial

.. toctree::
   :maxdepth: 4

   eemeter

Installation
------------

Execute the following command to install the eemeter package and its dependencies::

   pip install eemeter

Testing
-------

This library uses the py.test framework. To develop locally, clone the repo,
and in a virtual environment execute the following commands::

    $ git clone https://github.com/impactlab/eemeter
    $ cd eemeter
    $ mkvirtualenv eemeter
    $ pip install numpy scipy pytest lxml python-dateutil pandas xlrd sqlalchemy psycopg2
    $ python setup.py develop
    $ py.test --runslow

If you run into problems with the py.test executable, please ensure that you
are using the virtualenv py.test::

    $ py.test --version

Some tests are slow and are skipped by default; to run these (you should!),
use the `--runslow` flag::

    $ py.test --runslow

References
----------

- `PRISM <http://www.marean.mycpanel.princeton.edu/~marean/images/prism_intro.pdf>`_
- `ANSI/BPI-2400-S-2012 <http://www.bpi.org/Web%20Download/BPI%20Standards/BPI-2400-S-2012_Standard_Practice_for_Standardized_Qualification_of_Whole-House%20Energy%20Savings_9-28-12_sg.pdf>`_
- `NREL's Uniform Methods <http://energy.gov/sites/prod/files/2013/11/f5/53827-8.pdf>`_

Licence
-------

MIT
