Weather Data Caching
--------------------

In order to avoid putting an unnecessary load on external weather
sources, weather data is cached by default using json in a directory
:code:`~/.eemeter/cache`. The location of the directory can be changed by
setting:

.. code-block:: bash

    $ export EEMETER_WEATHER_CACHE_DIRECTORY=<full path to directory>
