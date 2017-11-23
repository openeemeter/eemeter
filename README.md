Open Energy Efficiency Meter 
============================

![Build status](https://travis-ci.org/openeemeter/eemeter.svg?branch=develop)
[![Coverage Status](https://coveralls.io/repos/github/openeemeter/eemeter/badge.svg?branch=develop)](https://coveralls.io/github/openeemeter/eemeter?branch=develop)

Documentation
-------------

See [openee.io](https://www.openee.io/open-source/get-started).

Dev Installation
----------------

    $ git clone https://github.com/openeemeter/eemeter
    $ cd eemeter
    $ mkvirtualenv eemeter
    (eemeter)$ pip install -e .
    (eemeter)$ pip install -r dev_requirements.txt
    (eemeter)$ workon # gives you access to virtualenv py.test executable

Testing
-------

This library uses the py.test framework. To develop locally, clone the repo,
and in a virtual environment execute the following commands:

    $ py.test

If you run into problems with the py.test executable, please ensure that you
are using the virtualenv py.test:

    $ py.test --version

Licence
-------

MIT
