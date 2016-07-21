Testing
-------

This library uses the py.test framework. To develop locally, clone the repo,
and in a virtual environment execute the following commands:

.. code-block:: bash

    $ git clone https://github.com/impactlab/eemeter
    $ cd eemeter
    $ mkvirtualenv eemeter
    $ pip install -r dev_requirements.txt
    $ pip install -e .
    $ tox
