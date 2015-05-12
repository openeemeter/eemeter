Tutorial
========

Introduction
------------

The Open Energy Efficiency Meter is an engine for measurement and verification
of energy savings across and between energy efficiency programs for both
commercial and residential applications. At its core, this package is a
framework for specifying how efficiency metrics should be calculated from raw
consumption data. It is flexible enough to meet the needs of different program
implementers without sacrificing the ability to standardize on particular
models and methods, and adds the ability to generate realtime or near realtime
reports on realized energy savings.

The meter handles import of consumption data from various formats and
data-stores, including Green Button, HPXML, and SEED. It handles downloading
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

Using an existing meter
-----------------------

This tutorial will walk through how to use an existing meter to evaluate the
energy consumption of a portfolio of buildings.

For this tutorial, we'll use sample data, but please see below for a
tutorial on connecting to a database, importing Green Button XML, or importing
Home Performance XML.

We will start by creating a portfolio by specifying distributions to draw
parameters for simple temperature sensitivity models of electricity and
natural gas consumption.

The following parameter distributions are for generating fake data using
a model which takes both heating degree days (HDD) and cooling degree
days (CDD) into account. This is a suitable model for monthly electricity
consumption.

.. code-block:: python

    from eemeter.models import TemperatureSensitivityModel
    from scipy.stats import uniform

    electricity_consumption_model = TemperatureSensitivityModel(heating=True,cooling=True)

    electricity_param_distributions = {
        "cooling_slope": uniform(loc=1, scale=.5), # consumption per CDD
        "heating_slope": uniform(loc=1, scale=.5), # consumption per HDD
        "base_consumption": uniform(loc=5, scale=5), # per day
        "cooling_reference_temperature": uniform(loc=70, scale=5), # degF
        "heating_reference_temperature": uniform(loc=60, scale=5) # degF
    }
    electricity_param_delta_distributions = {
        "cooling_slope": uniform(loc=-.2, scale=.3), # change in HDD temperature sensitivity post retrofit
        "heating_slope": uniform(loc=-.2, scale=.3), # change in HDD temperature sensitivity post retrofit
        "base_consumption": uniform(loc=-2, scale=3), # change in base load post retrofit
        "cooling_reference_temperature": uniform(loc=0, scale=0), # no change
        "heating_reference_temperature": uniform(loc=0, scale=0) # no change
    }

The following parameter distributions are for generating fake data using
a model which takes only heating degree days (HDD) into account. This is
a suitable model for monthly natural gas consumption.

.. code-block:: python

    gas_consumption_model = TemperatureSensitivityModel(heating=True,cooling=False)

    gas_param_distributions = {
            "heating_slope": uniform(loc=1, scale=.5),
            "base_consumption": uniform(loc=5, scale=5),
            "heating_reference_temperature": uniform(loc=60, scale=5)
    }
    gas_param_delta_distributions = {
            "heating_slope": uniform(loc=-.2, scale=.3),
            "base_consumption": uniform(loc=-2, scale=3),
            "heating_reference_temperature": uniform(loc=0, scale=0)
    }

With models and parameter distributions picked out, we can create a
ProjectGenerator from which we can create portfolios of projects.

.. code-block:: python

    from eemeter.generator import ProjectGenerator

    generator = ProjectGenerator(electricity_consumption_model,
                                 gas_consumption_model,
                                 electricity_param_distributions,
                                 electricity_param_delta_distributions,
                                 gas_param_distributions,
                                 gas_param_delta_distributions)

To make this generator work, we must provide it with weather data and usage
periods. Here, we create weather sources which automatically fetch data from
O'Hare INTL Airport near Chicago, IL. Fetch data (which can and should be
cached - see below) by providing the USAF weather station identifier
corresponding to the station.

.. code-block:: python

    from eemeter.weather import GSODWeatherSource
    from eemeter.weather import TMY3WeatherSource

    from datetime import datetime

    start_date = datetime(2012,1,1)

    ohare_weather_station_id = "725347" # Chicago O'Hare Intl Airport

    weather_source = GSODWeatherSource(ohare_weather_station_id,start_date.year,datetime.now().year)
    weather_normal_source = TMY3WeatherSource(ohare_weather_station_id)

With weather sources and weather normal sources, we are now equipped to
generate some projects. We do this by picking sets of periods of time each
approximately one month long, and using weather data to simulate usage data
according to the models we picked above. (The project generator takes care of
the details of this). The project generator also takes retrofit start and
completion dates into account in order to simulate the effect of installing
a set of energy efficiency measures. In this case, we generate a small set of
10 projects.

.. code-block:: python

    from eemeter.consumption import ConsumptionHistory
    from eemeter.generator import generate_periods

    from datetime import timedelta
    import random

    n_projects = 10
    n_days = (datetime.now() - start_date).days

    project_data = []
    for _ in range(n_projects):

        #generate random monthly periods to treat as billing periods
        elec_periods = generate_periods(start_date,datetime.now())
        gas_periods = generate_periods(start_date,datetime.now())

        # pick retrofit dates somewhere in the right range
        retrofit_start_date = start_date + timedelta(days=random.randint(100,n_days-130))
        retrofit_completion_date = retrofit_start_date + timedelta(days=30)

        # generate consumption data that mimics applying a measure and seeing a decrease in energy use
        result = generator.generate(weather_source,
                                    weather_normal_source,
                                    elec_periods,
                                    gas_periods,
                                    retrofit_start_date,
                                    retrofit_completion_date)

        consumptions = result["electricity_consumptions"] + result["natural_gas_consumptions"]

        data = {"consumption_history": ConsumptionHistory(consumptions),
                "retrofit_start_date": retrofit_start_date,
                "retrofit_completion_date":retrofit_completion_date,
                "estimated_elec_savings": result["electricity_estimated_savings"],
                "estimated_gas_savings": result["natural_gas_estimated_savings"]}
        project_data.append(data)

Phew! All of that was just to generate some projects so that we could learn how
to use the core metering functions of the eemeter package.

Running the energy efficiency meter is actually quite simple: First, a meter
is instantitated; here we're using a simple PRISM implementation which requires
no initialization parameters. Next, the efficiency meter is run by supplying
the necessary inputs. Note that the function :code:`meter.get_inputs()` will
expose the structure of the meter and the inputs needed to run it.

.. code-block:: python

    from eemeter.meter import PRISMMeter

    meter = PRISMMeter()

    for project in project_data:

        ch = project["consumption_history"]
        ch_pre = ch.before(project["retrofit_start_date"])
        ch_post = ch.after(project["retrofit_completion_date"])

        result_pre = meter.evaluate(consumption_history=ch_pre,
                                weather_source=weather_source,
                                weather_normal_source=weather_normal_source)

        result_post = meter.evaluate(consumption_history=ch_post,
                                weather_source=weather_source,
                                weather_normal_source=weather_normal_source)


        actual_e = result_pre["annualized_usage_electricity"] - result_post["annualized_usage_electricity"]
        predicted_e = project["estimated_elec_savings"]

        actual_g = result_pre["annualized_usage_natural_gas"] - result_post["annualized_usage_natural_gas"]
        predicted_g = project["estimated_gas_savings"]

        print("Electricity savings actual//predicted (# bills [pre]-[post]): {:.02f} // {:.02f} ({}-{})"
                .format(actual_e,predicted_e,len(ch_pre.electricity),len(ch_post.electricity)))
        print("Natural gas savings actual//predicted (# bills [pre]-[post]): {:.02f} // {:.02f} ({}-{})"
                .format(actual_g,predicted_g,len(ch_pre.natural_gas),len(ch_post.natural_gas)))
        print("")

This will print something like the following::

    Electricity savings actual//predicted (# bills [pre]-[post]): 1358.27 // 1358.27 (10-27)
    Natural gas savings actual//predicted (# bills [pre]-[post]): 1625.46 // 1625.46 (10-28)

    Electricity savings actual//predicted (# bills [pre]-[post]): 149.83 // 98.67 (13-22)
    Natural gas savings actual//predicted (# bills [pre]-[post]): 517.03 // 517.03 (14-22)

        :
        :
        :

    Electricity savings actual//predicted (# bills [pre]-[post]): 563.16 // 563.16 (20-16)
    Natural gas savings actual//predicted (# bills [pre]-[post]): -483.50 // -483.50 (20-16)

That's it! The results from all meters are python dictionaries keyed by strings.
Read on to learn how to load and stream your own data, or create your own
meters.

Loading consumption data
------------------------

To load consumption data, you'll need to use the SEED importer, the
HPXML importer, the Green Button XML importer, or initialize
the objects yourself. See :ref:`eemeter-importers` for usage details.

Consumption data consists of a quantity of energy (as defined by a magnitude a
physical unit) of a particular fuel type consumed during a time period (as
defined by start and end datetime objects). Additionally, a consumption data
point may also indicate that it was estimated, as some meters require this bit
of information for additional accuracy.

A collection of Consumption data related to a single project is grouped into a
ConsumptionHistory object, which helps keep the data organized by time period
and fuel type.

Here's a simple example of creating Consumption data from scratch, given two
lists of bills, one for electricity Jan-Dec 2014, one for natural gas Jan-Dec
2014.

.. code-block:: python

    from eemeter.consumption import Consumption
    from eemeter.consumption import ConsumptionHistory
    from datetime import datetime
    from calendar import monthrange

    kwh_electricity = [123,412,523,238,239,908,986,786,256,463,102,122]
    thm_natural_gas = [241,143,178,78,67,23,14,33,12,23,234,222]

    consumptions = []
    for i,(elec,gas) in enumerate(zip(kwh_electricity,thm_natural_gas)):
        month = i + 1
        start_datetime = datetime(2014,month,1)
        end_datetime = datetime(2014,month,monthrange(2014,month)[1])
        elec_consumption = Consumption(elec,"kWh","electricity",start_datetime,end_datetime,estimated=False)
        gas_consumption = Consumption(gas,"therm","natural_gas",start_datetime,end_datetime,estimated=False)
        consumptions.append(elec_consumption)
        consumptions.append(gas_consumption)

    consumption_history = ConsumptionHistory(consumptions)

Consumption energy data is stored internally in Joules, so to access it, you
must also supply the unit you are interested in.

.. code-block:: python

    >>> consumption_history.electricity[0].kWh
    123.00000000000001

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

    meter = DummyMeter()
    result = meter.evaluate(value=10)

.. code-block:: python

    from eemeter.config.yaml_parser import load

    meter_yaml = "!obj:eemeter.meter.DummyMeter {}"
    meter = load(meter_yaml)
    result = meter.evaluate(value=10)

In the example above, it's clearly more straightforward to directly declare the
meter using python. However, since meters are so hierarchical, a specification
like the following is usually more readable and straightforward. Note the usage
of structural helper meters like :code:`Sequence` and
:code:`Condition`, which allow for more flexible meter component
definitions.

.. code-block:: python

    prism_meter_yaml = """
        !obj:eemeter.meter.Sequence {
            sequence: [
                !obj:eemeter.meter.FuelTypePresenceMeter {
                    fuel_types: [electricity,natural_gas]
                },
                !obj:eemeter.meter.Condition {
                    condition_parameter: electricity_presence,
                    success: !obj:eemeter.meter.Sequence {
                        sequence: [
                            !obj:eemeter.meter.TemperatureSensitivityParameterOptimizationMeter {
                                fuel_unit_str: "kWh",
                                fuel_type: "electricity",
                                temperature_unit_str: "degF",
                                model: !obj:eemeter.models.TemperatureSensitivityModel &elec_model {
                                    cooling: True,
                                    heating: True,
                                    initial_params: {
                                        base_consumption: 0,
                                        heating_slope: 0,
                                        cooling_slope: 0,
                                        heating_reference_temperature: 60,
                                        cooling_reference_temperature: 70,
                                    },
                                    param_bounds: {
                                        base_consumption: [-20,80],
                                        heating_slope: [0,5],
                                        cooling_slope: [0,5],
                                        heating_reference_temperature: [58,66],
                                        cooling_reference_temperature: [64,72],
                                    },
                                },
                            },
                            !obj:eemeter.meter.AnnualizedUsageMeter {
                                fuel_type: "electricity",
                                temperature_unit_str: "degF",
                                model: *elec_model,
                            },
                        ],
                        output_mapping: {
                            temp_sensitivity_params: temp_sensitivity_params_electricity,
                            annualized_usage: annualized_usage_electricity,
                            daily_standard_error: daily_standard_error_electricity,
                        },
                    },
                },
                !obj:eemeter.meter.Conditionr {
                    condition_parameter: natural_gas_presence,
                    success: !obj:eemeter.meter.Sequence {
                        sequence: [
                            !obj:eemeter.meter.TemperatureSensitivityParameterOptimizationMeter {
                                fuel_unit_str: "therms",
                                fuel_type: "natural_gas",
                                temperature_unit_str: "degF",
                                model: !obj:eemeter.models.TemperatureSensitivityModel &gas_model {
                                    cooling: False,
                                    heating: True,
                                    initial_params: {
                                        base_consumption: 0,
                                        heating_slope: 0,
                                        heating_reference_temperature: 60,
                                    },
                                    param_bounds: {
                                        base_consumption: [0,10],
                                        heating_slope: [0,5],
                                        heating_reference_temperature: [58,66],
                                    },
                                },
                            },
                            !obj:eemeter.meter.AnnualizedUsageMeter {
                                fuel_type: "natural_gas",
                                temperature_unit_str: "degF",
                                model: *gas_model,
                            },
                        ],
                        output_mapping: {
                            temp_sensitivity_params: temp_sensitivity_params_natural_gas,
                            annualized_usage: annualized_usage_natural_gas,
                            daily_standard_error: daily_standard_error_natural_gas,
                        },
                    },
                },
            ]
        }
    """
    meter = load(prism_meter_yaml)
    result = meter.evaluate(consumption_history=...,
                            weather_source=...,
                            weather_normal_source=...)

Another benefit to using structured YAML for meter specification is that the
meter specifications can be stored externally as readable text files.

Caching Weather Data
--------------------

If you would like to cache weather data, please install :code:`sqlalchemy` and
set the following environment variable, which must contain the credentials to
a database you have set up for caching. If this variable is set properly, it
will cache weather as it is pulled from various sources::

    export EEMETER_WEATHER_CACHE_DATABASE_URL=dbtype://username:password@host:port/dbname

For additional information on the syntax of the url, please see sqlalchemy docs.

Creating a Weather Source from WeatherSourceBase
------------------------------------------------

Occasionally, you may want to incorporate a weather source of your own. To do
this, it is often easiest to extend the API by inheriting from the class
:code:`eemeter.meter.WeatherSourceBase`. To do this, you need only define the
method


    class MyWeatherSource(WeatherSourceBase):

        def get_internal_unit_daily_average_temperature(self,day):
            # return the average temperature on the given day, according to
            # your weather source. Use source units.

If you are defining a weather normal source, add the `WeatherNormalMixin`.

If you wish to take advantage of the caching mechanisms provided by `eemeter`,
use a `CachedDataMixin`.

