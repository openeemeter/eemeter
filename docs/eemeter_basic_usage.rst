Basic Usage
-----------

This tutorial will cover the three basic steps for using the :code:`eemeter`
package:

1. :ref:`data prepartation <data-preparation>`
2. :ref:`running meters <running-meters>`
3. :ref:`inspecting results <inspecting-results>`

This tutorial is also available as a jupyter :download:`notebook <eemeter_basic_usage.ipynb>`:

Before getting started, download some sample energy data and project data:

- energy data :download:`CSV <sample-energy-data_project-ABC_zipcode-50321.csv>`

- project data :download:`CSV <sample-project-data.csv>`

This sample data was created using
:download:`this <eemeter_basic_usage_data_creation.ipynb>` jupyter notebook
which you should reference if you have questions about the data.

.. note::

    Most users of the EEmeter stack do not directly use the :code:`eemeter`
    package for loading their data. Instead, they use the :ref:`datastore`,
    which uses the eemeter internally. To learn to use the datastore, head
    over to :ref:`this tutorial <datastore-basic-usage>`.

.. _data-preparation:

Data preparation
^^^^^^^^^^^^^^^^

The basic container for project data is the :code:`eemeter.structures.Project`
object. This object contains all of the data necessary for running a meter.

There are three items it requires:

    1. An :code:`EnergyTraceSet`, which is a collection of :code:`EnergyTrace` s
    2. An :code:`list` of :code:`Intervention` s
    3. An :code:`eemeter.structures.ZIPCodeSite`

Let's start by creating an :code:`EnergyTrace`. Internally, :code:`EnergyTrace`
objects use `numpy <http://docs.scipy.org/doc/numpy/reference/>`_ and
`pandas <http://pandas.pydata.org/pandas-docs/stable/>`_, which are nearly
ubiquitous python packages for efficient numerical computation and
data analysis, respectively.


Since this data is not in a format eemeter recognizes, we need to load it.
Let's load this data using a parser we create to turn this data into a
format that eemeter recognizes.

We will load data from formatted records using an
`eemeter.io.serializer.ArbitraryStartSerializer`.

.. code-block:: python

    # library imports
    from eemeter.structures import (
        EnergyTrace,
        EnergyTraceSet,
        Intervention,
        ZIPCodeSite,
        Project
    )
    from eemeter.io.serializers import ArbitraryStartSerializer
    from eemeter.ee.meter import EnergyEfficiencyMeter
    import pandas as pd
    import pytz

First, we import the energy data from the sample CSV and transform it into
records.

.. code-block:: python

    energy_data = pd.read_csv(
        'sample-energy-data_project-ABC_zipcode-50321.csv',
        parse_dates=['date'], dtype={'zipcode': str})

    records = [{
        "start": pytz.UTC.localize(row.date.to_datetime()),
        "value": row.value,
        "estimated": row.estimated,
    } for _, row in energy_data.iterrows()]

The records we just created look like this:

.. code-block:: python

    >>> records
    [
        {
            'estimated': False,
            'start': datetime.datetime(2011, 1, 1, 0, 0, tzinfo=<UTC>),
            'value': 57.8
        },
        {
            'estimated': False,
            'start': datetime.datetime(2011, 1, 2, 0, 0, tzinfo=<UTC>),
            'value': 64.8
        },
        {
            'estimated': False,
            'start': datetime.datetime(2011, 1, 3, 0, 0, tzinfo=<UTC>),
            'value': 49.5
        },
        ...
    ]

Next, we load our records into an :code:`EnergyTrace`. We give it units
:code:`"kWh"` and interpretation :code:`"ELECTRICITY_CONSUMPTION_SUPPLIED"`,
which means that this is electricity consumed by the building and supplied by
a utility (rather than by solar panels or other on-site generation).
We also pass in an instance of the record serializer
:code:`ArbitraryStartSerializer` to show it how to interpret the records.

.. code-block:: python

    energy_trace = EnergyTrace(
        records=records,
        unit="KWH",
        interpretation="ELECTRICITY_CONSUMPTION_SUPPLIED",
        serializer=ArbitraryStartSerializer())

The energy trace data looks like this:

.. code-block:: python

    >>> energy_trace.data[:3]
                               value estimated
    2011-01-01 00:00:00+00:00   57.8     False
    2011-01-02 00:00:00+00:00   64.8     False
    2011-01-03 00:00:00+00:00   49.5     False

Though we only have one trace here, we will often have more than one trace.
Because of that, projects expect an :code:`EnergyTraceSet`, which is a labeled
set of :code:`EnergyTrace` objects. We give it the
:code:`trace_id` supplied in the CSV.

.. code-block:: python

    energy_trace_set = EnergyTraceSet([energy_trace], labels=["DEF"])

Now we load the rest of the project data from the sample project data CSV.
This CSV includes the project_id (Which we don't use in this tutorial), the
ZIP code of the building, and the dates retrofit work for this project started
and completed.

.. code-block:: python

    project_data = pd.read_csv(
        'sample-project-data.csv',
        parse_dates=['retrofit_start_date', 'retrofit_end_date']).iloc[0]

We create an :code:`Intervention` from the retrofit start and end dates and
wrap it in a list:

.. code-block:: python

    retrofit_start_date = pytz.UTC.localize(project_data.retrofit_start_date)
    retrofit_end_date = pytz.UTC.localize(project_data.retrofit_end_date)

    interventions = [Intervention(retrofit_start_date, retrofit_end_date)]

Then we create a :code:`ZIPCodeSite` for the project by passing in the zipcode:

.. code-block:: python

    site = ZIPCodeSite(project_data.zipcode)

Now we can create a project using the data we've loaded:

.. code-block:: python

    project = Project(energy_trace_set=energy_trace_set,
                      interventions=interventions,
                      site=site)

This completes the :code:`eemeter` data loading process.

.. _running-meters:

Running meters
^^^^^^^^^^^^^^

To run the EEmeter on the project, instantiate an :code:`EnergyEfficiencyMeter`
and run the :code:`.evaluate(project)` method, passing in the project we just
created:

.. code-block:: python

    meter = EnergyEfficiencyMeter()
    results = meter.evaluate(project)

That's it! Now we can inspect and use our results.

.. _inspecting-results:

Inspecting Results
^^^^^^^^^^^^^^^^^^

Let's quickly look through the results object so that we can understand what
they mean. The results are embedded in a nested python :code:`dict`:

.. code-block:: python

    >>> results
    {
        'weather_normal_source': TMY3WeatherSource("725460"),
        'weather_source': ISDWeatherSource("725460"),
        'modeling_period_set': ModelingPeriodSet(),
        'modeled_energy_traces': {
            'DEF': SplitModeledEnergyTrace()
        },
        'modeled_energy_trace_derivatives': {
            'DEF': {
                ('baseline', 'reporting'): {
                    'BASELINE': {
                        'annualized_weather_normal': (11051.638608992347, 142.473017350216, 156.41867795302684, 365),
                        'gross_predicted': (31806.370855869744, 251.56911436695583, 276.19340851303582, 1138)
                    },
                    'REPORTING': {
                        'annualized_weather_normal': (8758.2778181960675, 121.92101539941024, 137.24631002750746, 365),
                         'gross_predicted': (25208.101373932539, 215.27979428803133, 242.34015188210202, 1138)
                    }
                }
            }
        },
        'project_derivatives': {
            ('baseline', 'reporting'): {
                'ALL_FUELS_CONSUMPTION_SUPPLIED': {
                    'BASELINE': {
                        'annualized_weather_normal': (11051.638608992347, 142.473017350216, 156.41867795302684, 365),
                        'gross_predicted': (31806.370855869744, 251.56911436695583, 276.19340851303582, 1138)
                    },
                    'REPORTING': {
                        'annualized_weather_normal': (8758.2778181960675, 121.92101539941024, 137.24631002750746, 365),
                        'gross_predicted': (25208.101373932539, 215.27979428803133, 242.34015188210202, 1138)
                    }
                },
                'ELECTRICITY_CONSUMPTION_SUPPLIED': {
                    'BASELINE': {
                        'annualized_weather_normal': (11051.638608992347, 142.473017350216, 156.41867795302684, 365),
                        'gross_predicted': (31806.370855869744, 251.56911436695583, 276.19340851303582, 1138)
                    },
                    'REPORTING': {
                        'annualized_weather_normal': (8758.2778181960675, 121.92101539941024, 137.24631002750746, 365),
                        'gross_predicted': (25208.101373932539, 215.27979428803133, 242.34015188210202, 1138)
                    }
                },
                'ELECTRICITY_ON_SITE_GENERATION_UNCONSUMED': None,
                'NATURAL_GAS_CONSUMPTION_SUPPLIED': None
            }
        },
    }

Note the contents of the dictionary:

- :code:`'weather_source'`: An instance of
  :code:`eemeter.weather.ISDWeatherSource`.  The weather source used to
  gather observed weather data. The station at which this weather was
  recorded can be found by inspecting :code:`weather_source.station`.
  (Matched by ZIP code)
- :code:`'weather_normal_source'`: An instance of
  :code:`eemeter.weather.TMY3WeatherSource`. The weather normal source used
  to gather weather normal data. The station at which this weather normal
  data was recorded can be found by inspecting
  :code:`weather_normal_source.station`. (Matched by ZIP code)
- :code:`'modeling_period_set'`: An instance of
  :code:`eemeter.structures.ModelingPeriodSet`. The modeling periods
  determined by the intervention start and end dates; includes groupings.
  The default grouping for a single intervention is into two modeling
  periods called "baseline" and "reporting".
- :code:`'modeled_energy_traces'`: :code:`SplitModeledEnergyTraces` instances
  keyed by :code:`trace_id` (as given in the :code:`EnergyTraceSet`; includes
  models and fit statistics for each modeling period.
- :code:`'modeled_energy_trace_derivatives'`: energy results specific to each
  modeled energy trace, organized by trace_id and modeling period group.
- :code:`'project_derivatives'`: Project-level results which are aggregated up
  from the :code:`'modeled_energy_trace_derivatives'`.

The project derivatives are nested quite deeply. The nesting of key-value pairs
is as follows:

- 1st layer: Modeling Period Set id: a tuple of 1 baseline period id and 1
  reporting period id, usually :code:`('baseline', 'reporting')` -
  contains the results specific to this pair of modeling periods.
- 2nd layer: Trace interpretation: a string describing the trace
  interpretation; in our case :code:`"ELECTRICITY_CONSUMPTION_SUPPLIED"`
- 3rd layer: :code:`'BASELINE'` and :code:`'REPORTING'` - these are fixed
  labels that always appear at this level; they demarcate the baseline
  aggregations and the reporting aggregations.
- 4th layer: :ref:`'annualized_weather_normal' <glossary-annualized-weather-normal>`
  and :ref:`'gross_predicted' <glossary-gross-predicted>` - these are also
  fixed labels that always appear at this level to indicate the type of the savings values.

At the final layers are a 4-tuple of results
:code:`(value, lower, upper, n)`: :code:`value`, indicating the estimated
expected value of the selected result; :code:`lower`, a number which can
be subtracted from :code:`value` to obtain the lower 95% confidence
interval bound; :code:`upper`,  a number which can be added to
:code:`value` to obtain the upper 95% confidence interval bound, and
:code:`n`, the total number of records that went into calculation of
this value.

To obtain savings numbers, the reporting value should be subtracted from the
baseline value as described in :ref:`methods-overview`.

Let's select the most useful results from the eemeter, the project-level
derivatives. Note the modeling_period_set selector at the first level:
`('baseline', 'reporting')`


.. code-block:: python

    project_derivatives = results['project_derivatives']

.. code-block:: python

    >>> project_derivatives.keys()
    dict_keys([('baseline', 'reporting')])

.. code-block:: python

    modeling_period_set_results = project_derivatives[('baseline', 'reporting')]

Now we can select the desired interpretation; four are available.

.. code-block:: python

    >>> modeling_period_set_results.keys()
    dict_keys(['NATURAL_GAS_CONSUMPTION_SUPPLIED', 'ALL_FUELS_CONSUMPTION_SUPPLIED', 'ELECTRICITY_CONSUMPTION_SUPPLIED', 'ELECTRICITY_ON_SITE_GENERATION_UNCONSUMED'])

.. code-block:: python

    electricity_consumption_supplied_results = modeling_period_set_results['ELECTRICITY_CONSUMPTION_SUPPLIED']

The interpretation level results are broken into :code:`"BASELINE"` and
:code:`"REPORTING"` in all cases in which they are available; otherwise, the
value is `None`.)

.. code-block:: python

    >>> electricity_consumption_supplied_results.keys()
    dict_keys(['BASELINE', 'REPORTING'])

.. code-block:: python

    baseline_results = electricity_consumption_supplied_results["BASELINE"]
    reporting_results = electricity_consumption_supplied_results["REPORTING"]

These results have two components as well - the type of savings.

.. code-block:: python

    >>> baseline_results.keys()
    dict_keys(['gross_predicted', 'annualized_weather_normal'])
    >>> reporting_results.keys()
    dict_keys(['gross_predicted', 'annualized_weather_normal'])

We select the results for one of them:

.. code-block:: python

    baseline_normal = baseline_results['annualized_weather_normal']
    reporting_normal = reporting_results['annualized_weather_normal']

As described above, each energy value also includes upper and lower bounds,
but can also be used directly to determine savings.

.. code-block:: python

    percent_savings = (baseline_normal[0] - reporting_normal[0]) / baseline_normal[0]

.. code-block:: python

    >>> percent_savings
    0.20751319075256849

This percent savings value (~20%) is consistent with the savings created in the
fake data.
