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

Loading data from CSV
---------------------

To load data from a CSV, use :any:`eemeter.meter_data_from_csv`::

    >>> meter_data = eemeter.meter_data_from_csv(f)  # file handle

To load temperature data from a CSV, use :any:`eemeter.temperature_data_from_csv`.
(See also :any:`EEweather <eeweather:index>`)::

    >>> temperature_data = eemeter.temperature_data_from_csv(f)  # file handle

This also works with gzipped files (like the sample data)::

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
                                                                                   _
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


Running the CalTRACK methods
----------------------------

End-to-end running CalTRACK methods.

Using the CLI
-------------

More in-depth info on CLI usage

Visualization
-------------

Visualization tricks

Obtaining weather data
----------------------

Weather data can be obtained using the :any:`EEweather <eeweather:index>` package.

Using with anaconda
-------------------

Making installation easier for some folks.
