Installation
------------

.. note::

    If you are installing python for the first time, we recommend using
    Anaconda_, a free python distribution with builds for windows, mac os,
    and linux.

To get started with the eemeter, use pip::

    $ pip install eemeter

Make sure you have the latest version:

.. code-block:: python

    >>> import eemeter; eemeter.get_version()
    '1.4.0'

The `eemeter` package itself does not use C extensions. However, some eemeter
dependencies do. These can be a bit trickier to install. If issues arise when
pip installing eemeter, verify that the packges with C extensions are properly
installing. Specifically, verify that these installation commands complete
without errors::

    pip install lxml
    pip install numpy

If they fail, please see follow installation instructions for those packages
(lxml_, numpy_).

Some statsmodels installations require numpy to be installed. If you run into
errors with the statsmodels installation, be sure numpy is installed before
attempting to install statsmodels. Once statsmodels is installed correctly,
install eemeter.

.. _Anaconda: https://www.continuum.io/downloads
.. _lxml: http://lxml.de/installation.html
.. _numpy: http://scipy.org/install.html
