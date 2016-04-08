Tutorial
========

Introduction
------------

The Open Energy Efficiency Meter is an engine for measurement and verification
of energy savings across and between energy efficiency programs for both
commercial and residential applications, with a focus on computing energy
savings using the IPMVP Option (C) Whole Facility option.

At its core, this package is a framework for specifying how efficiency metrics
should be calculated from building-level energy consumption data. It is
flexible enough to meet the needs of different program implementers without
sacrificing the ability to standardize on particular models and methods, and
facilitates the growing need to generate realtime or near realtime reports
on realized energy savings.

The meter handles import of consumption data from various formats and
datastores, including Green Button and HPXML. It handles downloading
and local caching of realtime or near realtime weather data and normals,
including TMY3, GSOD, ISD, and wunderground.com.

Although the package comes with pre-written implementations of several
efficiency metering standards, custom meters can be written by assembling
existing meter components or by writing new components.

.. warning::

   The `eemeter` package is under rapid development; we are working quickly
   toward a stable release. In the mean time, please proceed to use the package,
   but as you do so, recognize that the API is in flux and the docs might not
   be up-to-date. Feel free to contribute changes or open issues on
   `github <https://github.com/impactlab/eemeter>`_ to report bugs, request
   features, or make suggestions.

Installation
------------

To get started with the eemeter, use pip::

    $ pip install eemeter

Make sure you have the latest version:

.. code-block:: python

    >>> import eemeter; eemeter.get_version()
    '0.3.12'

Using an existing meter
-----------------------

This tutorial will walk through how to use an existing meter to evaluate the
energy consumption of a portfolio of buildings.

For this tutorial, we'll use sample data, but please see below for a
tutorial on connecting to a database, importing Green Button XML, or importing
Home Performance XML.

First, some imports.

.. code-block:: python

    from eemeter.consumption import ConsumptionData
    from eemeter.examples import get_example_project
    from eemeter.meter import DefaultResidentialMeter
    from eemeter.meter import DataCollection

All we'll need to get started is a project, which is an association building
data, retrofit dates, and weather data.

We can initialize a sample by passing in a zipcode, e.g.:

.. code-block:: python

    project = get_example_project("94087")

This project contains location information, usage data from gas and electricity
bills, and baseline and reporting periods. In this case, we imagine a
retrofit to have happenend on Jan. 1, 2013. We have data from 2 years before
and after this retrofit. By the way, we probably could have gotten by with just
a year pre/post, but don't worry! - the default meter will make sure there's
enough data for a statistically significant result and will flag the result if
there's not.

.. code-block:: python

    meter = DefaultResidentialMeter()

To run the meter, we first embed our project into a DataCollection, which
is used as input to the meter, then we pass the data collection to the meter!
Here it is as a one-liner:

.. code-block:: python

    results = meter.evaluate(DataCollection(project=project))

The meter results (there are many!) contain annualized usage metrics from
before and after the retrofit, in the baseline and reporting periods. We can
examine these results to obtain savings estimates.

.. code-block:: python

    electricity_usage_pre = results.get_data("annualized_usage", ["electricity", "baseline"]).value
    electricity_usage_post = results.get_data("annualized_usage", ["electricity", "reporting"]).value
    natural_gas_usage_pre = results.get_data("annualized_usage", ["natural_gas", "baseline"]).value
    natural_gas_usage_post = results.get_data("annualized_usage", ["natural_gas", "reporting"]).value

    electricity_savings = (electricity_usage_pre - electricity_usage_post) / electricity_usage_pre
    natural_gas_savings = (natural_gas_usage_pre - natural_gas_usage_post) / natural_gas_usage_pre

Now we can inspect our results:

.. code-block:: python

    >>> electricity_savings
    0.50061411300996794
    >>> natural_gas_savings
    0.50139379943863116

If you prefer, you can also look serialized json data from your meter run:

.. code-block:: python

    json_data = results.json()


Loading consumption data
------------------------

Consumption data consists of a quantity of energy (as defined by a magnitude a
physical unit) of a particular fuel type consumed during a time period (as
defined by start and end datetime objects). Additionally, a consumption data
point may also indicate that it was estimated, as some meters require this bit
of information for additional accuracy.

To load consumption data, you'll need to
import from Green Button XML (see :ref:`eemeter-parsers`),
or load objects yourself (see :ref:`eemeter-consumption`).

Creating a custom meter
-----------------------

Meters can be defined from scratch or customized to meet specific needs. For
instance, a particular user might want to incorporate unique data quality flags,
and another user might want to optimize evaluation for a particular parallel
computing environment.

Meters are modular, hierarchical and swappable; often the most convenient
and readable way to define them is to use YAML, as we will do here. Note that
the particular YAML format we use here has been customized (ht: pylearn2_) with
an :code:`!obj` tag to automate python object specification. Note that JSON is
always valid YAML.

.. _pylearn2: http://deeplearning.net/software/pylearn2/

Consider the following equivalent examples, which both declare a "dummy" meter
that simply spits out or renames the input values. The first loads the
meter as usual; the second declares an equivalent meter using YAML, then loads
the result.

.. code-block:: python

    from eemeter.meter import DummyMeter
    from eemeter.meter import DataCollection

    meter = DummyMeter()
    data_collection = DataCollection(value=10)
    result = meter.evaluate(data_collection)

.. code-block:: python

    from eemeter.config.yaml_parser import load

    meter_yaml = "!obj:eemeter.meter.DummyMeter {}"
    meter = load(meter_yaml)
    data_collection = DataCollection(value=10)
    result = meter.evaluate(data_collection)

In the example above, it's clearly more straightforward to directly declare the
meter using python. However, since meters are so hierarchical, a specification
like the following is usually more readable and straightforward. Note the usage
of control flow meters (see :ref:`eemeter-meter-control`) like :code:`Sequence`
and :code:`Condition`, which allow for more flexible meter component
definitions.

Please see the default meter implementation for an example of YAML meter
specification (:ref:`eemeter-meter-default`).

One benefit to using structured YAML for meter specification is that the
meter specifications can be stored externally as readable text files.

Weather Data Caching
--------------------

In order to avoid putting an unnecessary load on external weather
sources, weather data is cached by default using json in a directory
`~/.eemeter/cache`. The location of the directory can be changed by setting::

    export EEMETER_WEATHER_CACHE_DIRECTORY=<full path to directory>
