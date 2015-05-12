Open Energy Efficiency Meter
============================

Documentation
-------------

Docs on [RTD](http://eemeter.readthedocs.org/en/latest/).

Installation
------------

Execute the following command to install:

    $ pip install eemeter

Testing
-------

This library uses the py.test framework. To develop locally, clone the repo,
and in a virtual environment execute the following commands:

    $ git clone https://github.com/impactlab/eemeter
    $ cd eemeter
    $ mkvirtualenv eemeter
    $ pip install numpy scipy pytest lxml python-dateutil pandas xlrd sqlalchemy psycopg2
    $ python setup.py develop
    $ py.test --runslow

If you run into problems with the py.test executable, please ensure that you
are using the virtualenv py.test:

    $ `py.test --version`

Some tests are slow and are skipped by default; to run these (you should!),
use the `--runslow` flag:

    $ py.test --runslow

Licence
-------

MIT
