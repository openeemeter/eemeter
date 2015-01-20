Open Energy Efficiency Meter
============================

Documentation
-------------

Docs on [RTD](http://eemeter.readthedocs.org/en/latest/).

Installation
------------

Execute the following command to install:

    $ pip install git+git://github.com/philngo/ee-meter.git#egg=ee-meter

Testing
-------

This library uses the py.test framework. To develop locally, clone the repo,
and in a virtual environment execute the following commands:

    $ git clone https://github.com/philngo/ee-meter
    $ cd eemeter
    $ python setup.py develop
    $ pip install pytest
    $ py.test

You should ensure that you are using the py.test executable within the
virtualenv with `py.test --version`.

Some tests are slow and are skipped by default; to run these, use the `--runslow` flag:

    $ py.test --runslow

Contributors
------------

+ Phil Ngo
+ Matt Gee
+ Eric Potash

Licence
-------

MIT
