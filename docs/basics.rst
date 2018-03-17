Basic Usage
===========

Loading sample data
-------------------

EEMeter comes packages with some simulated sample data.

.. note::

    This data is not to be used for methods testing! It is designed to have
    obvious (but completely unrealistic) behavior to showcase building
    temperature response.

See a list of available sample data files, use :any:`eemeter.samples`::

    >>> eemeter.samples()
    ['il-electricity-cdd-hdd-hourly',
     'il-electricity-cdd-hdd-daily',
     'il-electricity-cdd-hdd-billing_monthly',
     'il-electricity-cdd-hdd-billing_bimonthly',
     'il-electricity-cdd-only-hourly',
     'il-electricity-cdd-only-daily',
     'il-electricity-cdd-only-billing_monthly',
     'il-electricity-cdd-only-billing_bimonthly',
     'il-gas-hdd-only-hourly',
     'il-gas-hdd-only-daily',
     'il-gas-hdd-only-billing_monthly',
     'il-gas-hdd-only-billing_bimonthly',
     'il-gas-intercept-only-hourly',
     'il-gas-intercept-only-daily',
     'il-gas-intercept-only-billing_monthly',
     'il-gas-intercept-only-billing_bimonthly']

Load meter data, temperature data, and metadata, use :any:`eemeter.load_sample`::

    >>> meter_data, temperature_data, metadata = \
    ...     eemeter.load_sample('il-electricity-cdd-hdd-daily')
    >>> meter_data.head()
                               value
    start
    2015-11-22 00:00:00+00:00  32.34
    2015-11-23 00:00:00+00:00  23.80
    2015-11-24 00:00:00+00:00  26.26
    2015-11-25 00:00:00+00:00  21.32
    2015-11-26 00:00:00+00:00   6.70
    >>> temperature_data.head()
    dt
    2015-11-22 06:00:00+00:00    21.01
    2015-11-22 07:00:00+00:00    20.35
    2015-11-22 08:00:00+00:00    19.38
    2015-11-22 09:00:00+00:00    19.02
    2015-11-22 10:00:00+00:00    17.82
    Name: tempF, dtype: float64

The metadata :any:`dict` contains simulated project ground truth, such as roughly
expected disaggregated annual usage, savings, and project dates.

Loading data from CSV
---------------------

Default meter data CSV format::

    start,value
    2015-11-22T00:00:00+00:00,32.34
    2015-11-23T00:00:00+00:00,23.80
    2015-11-24T00:00:00+00:00,26.26
    2015-11-25T00:00:00+00:00,21.32
    2015-11-26T00:00:00+00:00,6.70
    ...

To load meter data from a CSV, use :any:`eemeter.meter_data_from_csv`::

    >>> meter_data = eemeter.meter_data_from_csv(f)  # file handle

The :any:`eemeter.meter_data_from_csv` has lots of configurable options for
data that is formatted differently! Check out the API docs for more info.

Default temperature data CSV format::

    dt,tempF
    2015-11-22T00:00:00+06:00,21.01
    2015-11-22T01:00:00+06:00,20.35
    2015-11-22T02:00:00+06:00,19.38
    2015-11-22T03:00:00+06:00,19.02
    2015-11-22T04:00:00+06:00,17.82
    ...

To load temperature data from a CSV, use :any:`eemeter.temperature_data_from_csv`.
(See also :any:`EEweather <eeweather:index>`)::

    >>> temperature_data = eemeter.temperature_data_from_csv(f)  # file handle

The :any:`eemeter.temperature_data_from_csv` also has lots of configurable
options for data that is formatted differently! Check out the API docs for
more info.

These methods also work with gzipped files (e.g., the sample data)::

    >>> meter_data = eemeter.meter_data_from_csv(f, gzipped=True)

If frequency is known (``'hourly'``, ``'daily'``), this will load that data
with an index of the appropriate frequency. This helps the data formatting
methods do the right thing.

::

    >>> daily_meter_data = eemeter.meter_data_from_csv(f, freq='daily')

Creating design matrix datasets
-------------------------------

To merge temperature data with meter data, use :any:`eemeter.merge_temperature_data`::


    >>> meter_data, temperature_data, metadata = \
    ...     eemeter.load_sample('il-electricity-cdd-hdd-daily')
    >>> data = eemeter.merge_temperature_data(meter_data, temperature_data)

By default, this will give you a :any:`pandas.DataFrame` with two columns:
``meter_value`` and ``temperature_mean``::

    >>> data.head()
                               meter_value  temperature_mean
    2015-11-22 00:00:00+00:00        32.34         26.740000
    2015-11-23 00:00:00+00:00        23.80         38.831667
    2015-11-24 00:00:00+00:00        26.26         41.304583
    2015-11-25 00:00:00+00:00        21.32         49.198333
    2015-11-26 00:00:00+00:00         6.70         57.856667

Other options for constructing datasets are available, such as data quality::

    >>> data = eemeter.merge_temperature_data(
    ...     meter_data, temperature_data, temperature_mean=False,
    ...     data_quality=True)
    >>> data.head()
                               meter_value  temperature_not_null  temperature_null
    start
    2015-11-22 00:00:00+00:00        32.34                    18               0.0
    2015-11-23 00:00:00+00:00        23.80                    24               0.0
    2015-11-24 00:00:00+00:00        26.26                    24               0.0
    2015-11-25 00:00:00+00:00        21.32                    24               0.0
    2015-11-26 00:00:00+00:00         6.70                    24               0.0

TODO(philngo): more info here about options for hdd/cdd values.

Running the CalTRACK methods
----------------------------

End-to-end running CalTRACK methods.

To run the CalTRACK daily or billing methods, you need a :any:`pandas.DataFrame` with
the following columns:

- ``meter_value``: Daily average metered usage values for each point.
- ``cdd_<cooling_balance_point>``: Average period daily cooling degree days for
  a particular cooling balance point.
- ``hdd_<heating_balance_point>``: Average period daily heating degree days for
  a particular heating balance point.

For each balance point you want to include in the grid search, you must
provide a ``cdd_<>`` or ``hdd_<>`` column.

Armed with this DataFrame (:any:`eemeter.merge_temperature_data` is a utility
that simplifies the process of creating this DataFrame), you can use
:any:`eemeter.caltrack_method` or (TODO) to fit a model.

You may also wish to filter your data to a baseline period or a reporting
period. To do so, use :any:`eemeter.get_baseline_data` or
:any:`eemeter.get_reporting_data`. For example::

    >>> baseline_data = eemeter.get_baseline_data(data, end=<baseline end date>, max_days=365)

CalTRACK Daily Methods
----------------------

Running caltrack daily methods::

    >>> model_fit = eemeter.caltrack_method(data)

CalTRACK Billing Methods
------------------------

Running caltrack billing methods::

    >>> model_fit = eemeter.caltrack_method(data, use_billing_preset=True)

It is essential that the data used in the CalTRACK billing methods is
*average daily* period usage (UPDm) and degree day values.

Data with this property is created by default by the
:any:`eemeter.merge_temperature_data` method, but can be controlled explicitly
with the ``use_mean_daily_values`` flag of that method.


Using the CLI
-------------

More in-depth info on CLI usage.


Visualization
-------------

Plotting results and models.


Obtaining weather data
----------------------

Weather data can be obtained using the :any:`EEweather <eeweather:index>` package.


Using with anaconda
-------------------

Making installation easier for some folks.
