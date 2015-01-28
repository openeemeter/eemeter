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
including TMY3, GSOD, ISD, wunderground.com, degreedays.net.

Although the package comes with pre-written implementations of several
efficiency metering standards, custom meters can be written by assembling
existing meter components or by writing new components.

Installation
------------

To get started with the eemeter, use pip::

    $ pip install git+git://github.com/philngo/ee-meter.git#egg=ee-meter

or download it from github and install it using the setup.py::

    $ git clone git://github.com/philngo/ee-meter.git
    $ cd ee-meter/
    $ python setup.py install

Using an existing meter
-----------------------

This tutorial will walk through how to use an existing meter to evaluate the
energy consumption of a building with realistically but stochastically
generated building.

A consumption generator uses a model that you specify to
realistically generate consumption data given a particular weather pattern. In
this case, we'll make one for electricity and one for natural gas, specifying
parameters for the models and units for the calculations.

.. code-block:: python

    from eemeter.generator import ConsumptionGenerator

    elec_generator = ConsumptionGenerator("electricity", "kWh", "degF", 60, 1, 65, 1, 1)
    gas_generator = ConsumptionGenerator("natural_gas", "therms", "degF", 60, 2, 65, 2, 2)

To make these generators work, we must provide them with weather data and usage
periods. Here, we create weather sources with data from O'Hare INTL Airport
near Chicago, IL, by providing a station identifier.

.. code-block:: python

    from eemeter.weather import GSODWeatherSource
    from eemeter.weather import TMY3WeatherSource

    ohare_weather_station_id = "725347"

    # This directory must contain a file called "725347TY.csv"
    path_to_tmy3_directory = "path/to/folder/"

    # Collect data from 2012 to 2014, inclusive.
    # (It may take a moment to collect this data - weather-data caching is soon to come!)
    ohare_weather_source = GSODWeatherSource(ohare_weather_station_id, 2012, 2014)

    # Collect TMY3 weather normals
    ohare_weather_normals = TMY3WeatherSource(ohare_weather_station_id, path_to_tmy3_directory))

This function to makes some datetime periods that can be used as the billing
periods. Note that there will also be generators for data at higher sampling
rates, but for this example we will stick with monthly billing data.
(TODO - incorporate this into the code base)

.. code-block:: python

    from eemeter.consumption import DatetimePeriod

    import numpy as np

    from datetime import timedelta

    def generate_monthly_periods(n_periods, start_datetime, base_time_interval):
        last_datetime = start_datetime
        periods = []
        for i in np.random.randint(-2, 3, size=n_periods):
            new_datetime = last_datetime + timedelta(days=int(base_time_interval + i))
            periods.append(DatetimePeriod(last_datetime, new_datetime))
            last_datetime = new_datetime
        return periods

Let's grab 24 periods whose lengths all vary slightly and use these to generate
fake consumption data.

.. code-block:: python

    from eemeter.consumption import ConsumptionHistory

    from datetime import datetime

    periods = generate_monthly_periods(24, datetime(2012, 1, 1), 365/12.)
    elec_consumptions = elec_generator.generate(ohare_weather_source,periods)
    gas_consumptions = gas_generator.generate(ohare_weather_source,periods)
    consumptions = elec_consumptions + gas_consumptions
    consumption_history = ConsumptionHistory(consumptions)

This is the core of the code for running the meter. First, a meter is
instantitated; here we're using a simple PRISM implementation. Second, a few
parameters are passed to the meter for evaluation.

.. code-block:: python

    from eemeter.meter import PRISMMeter

    meter = PRISMMeter()

    result = meter.evaluate(consumption_history=consumption_history,
                            weather_source=ohare_weather_source,
                            weather_normal_source=ohare_weather_normals)

The variable :code:`result` will contain something like the following:

.. code-block:: python

    {'annualized_usage_electricity': 6662.5901982011483,
     'annualized_usage_natural_gas': 13325.180786132325,
     'electricity_presence': True,
     'natural_gas_presence': True,
     'temp_sensitivity_params_electricity': array([  1.        ,  18.25367178,  60.        ]),
     'temp_sensitivity_params_natural_gas': array([  1.        ,  36.50734462,  60.        ])}


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
(usually? always?) valid YAML.

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
of structural helper meters like :code:`SequentialMeter` and
:code:`ConditionalMeter`, which allow for more flexible meter component
definitions.

.. code-block:: python

    prism_meter_yaml = """
        !obj:eemeter.meter.SequentialMeter {
            sequence: [
                !obj:eemeter.meter.FuelTypePresenceMeter {
                    fuel_types: [electricity,natural_gas]
                },
                !obj:eemeter.meter.ConditionalMeter {
                    condition_parameter: electricity_presence,
                    success: !obj:eemeter.meter.SequentialMeter {
                        sequence: [
                            !obj:eemeter.meter.TemperatureSensitivityParameterOptimizationMeter {
                                fuel_unit_str: "kWh",
                                fuel_type: "electricity",
                                temperature_unit_str: "degF",
                                model: !obj:eemeter.models.PRISMModel &elec_model {
                                    x0: [1.,1.,60],
                                    bounds: [[0,100],[0,100],[55,65]],
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
                        },
                    },
                },
                !obj:eemeter.meter.ConditionalMeter {
                    condition_parameter: natural_gas_presence,
                    success: !obj:eemeter.meter.SequentialMeter {
                        sequence: [
                            !obj:eemeter.meter.TemperatureSensitivityParameterOptimizationMeter {
                                fuel_unit_str: "therms",
                                fuel_type: "natural_gas",
                                temperature_unit_str: "degF",
                                model: !obj:eemeter.models.PRISMModel &gas_model {
                                    x0: [1.,1.,60],
                                    bounds: [[0,100],[0,100],[55,65]],
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
                        },
                    },
                },
            ]
        }
    """
    meter = load(prism_meter_yaml)
    result = meter.evaluate(value=10)

Another benefit to using structured YAML for meter specification is that the
meter specifications can be stored externally as readable text files.
