Tutorial
========

.. note::

    This tutorial assumes you have a working knowledge of the pandas library, a key
    eemeter dependency. For eemeter installation instructions, see
    :ref:`installation`. If you're new to pandas, the :any:`10 minutes to pandas
    tutorial <pandas:10min>` is a good primer. We recommend reading that and then
    coming back here.

Outline
-------

This tutorial is a self-paced walkthrough of how to use the eemeter package. We'll
cover the following:


.. toctree::
   :maxdepth: 3

   tutorial

The tutorial demonstrates how to use the package to run the CalTRACK Hourly, Daily,
and Billing methods.

.. _quickstart:

Quickstart
----------

Some folks may just want to see the code all in one place. This code is explained
in more detail in the course of the tutorial below.  See also
:ref:`caltrack-billing-daily-api`.

.. _caltrack-billing-daily-quickstart:

Quickstart for CalTRACK Billing/Daily
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here's how to run the CalTRACK billing/daily model.  See also
:ref:`caltrack-billing-daily-api`::

    import eemeter

    meter_data, temperature_data, sample_metadata = (
        eemeter.load_sample("il-electricity-cdd-hdd-daily")
    )

    # The dates of an analysis "blackout" period during which a project was performed.
    # This is synonymous with the CalTRACK "Intervention period" (See CalTRACK 1.4.4)
    blackout_start_date = sample_metadata["blackout_start_date"]
    blackout_end_date = sample_metadata["blackout_end_date"]

    # get meter data suitable for fitting a baseline model
    baseline_meter_data, warnings = eemeter.get_baseline_data(
        meter_data, end=blackout_start_date, max_days=365
    )

    # create a design matrix (the input to the model fitting step)
    baseline_design_matrix = eemeter.create_caltrack_daily_design_matrix(
        baseline_meter_data, temperature_data,
    )

    # build a CalTRACK model
    baseline_model = eemeter.fit_caltrack_usage_per_day_model(
        baseline_design_matrix,
    )

    # get a year of reporting period data
    reporting_meter_data, warnings = eemeter.get_reporting_data(
        meter_data, start=blackout_end_date, max_days=365
    )

    # compute metered savings for the year of the reporting period we've selected
    metered_savings_dataframe, error_bands = eemeter.metered_savings(
        baseline_model, reporting_meter_data,
        temperature_data, with_disaggregated=True
    )

    # total metered savings
    total_metered_savings = metered_savings_dataframe.metered_savings.sum()

.. _caltrack-hourly-quickstart:

Quickstart for CalTRACK Hourly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

And here's now to run the CalTRACK hourly model. Again, this is explained in more
detail below. See also :ref:`caltrack-hourly-api`::

    import eemeter

    meter_data, temperature_data, sample_metadata = (
        eemeter.load_sample("il-electricity-cdd-hdd-hourly")
    )

    # the dates if an analysis "blackout" period during which a project was performed.
    blackout_start_date = sample_metadata["blackout_start_date"]
    blackout_end_date = sample_metadata["blackout_end_date"]

    # get meter data suitable for fitting a baseline model
    baseline_meter_data, warnings = eemeter.get_baseline_data(
        meter_data, end=blackout_start_date, max_days=365
    )

    # create a design matrix for occupancy and segmentation
    preliminary_design_matrix = (
        eemeter.create_caltrack_hourly_preliminary_design_matrix(
            baseline_meter_data, temperature_data,
        )
    )

    # build 12 monthly models - each step from now on operates on each segment
    segmentation = eemeter.segment_time_series(
        preliminary_design_matrix.index,
        'three_month_weighted'
    )

    # assign an occupancy status to each hour of the week (0-167)
    occupancy_lookup = eemeter.estimate_hour_of_week_occupancy(
        preliminary_design_matrix,
        segmentation=segmentation,
    )

    # assign temperatures to bins
    occupied_temperature_bins, unoccupied_temperature_bins = eemeter.fit_temperature_bins(
        preliminary_design_matrix,
        segmentation=segmentation,
        occupancy_lookup=occupancy_lookup,
    )

    # build a design matrix for each monthly segment
    segmented_design_matrices = (
        eemeter.create_caltrack_hourly_segmented_design_matrices(
            preliminary_design_matrix,
            segmentation,
            occupancy_lookup,
            occupied_temperature_bins,
            unoccupied_temperature_bins,
        )
    )

    # build a CalTRACK hourly model
    baseline_model = eemeter.fit_caltrack_hourly_model(
        segmented_design_matrices,
        occupancy_lookup,
        occupied_temperature_bins,
        unoccupied_temperature_bins,
    )

    # get a year of reporting period data
    reporting_meter_data, warnings = eemeter.get_reporting_data(
        meter_data, start=blackout_end_date, max_days=365
    )

    # compute metered savings for the year of the reporting period we've selected
    metered_savings_dataframe, error_bands = eemeter.metered_savings(
        baseline_model, reporting_meter_data,
        temperature_data, with_disaggregated=True
    )

    # total metered savings
    total_metered_savings = metered_savings_dataframe.metered_savings.sum()

.. _data-formats:

Data Formats
------------

The three essential inputs to eemeter library functions are the following:

1. :ref:`meter-data`
2. :ref:`temperature-data` from a nearby weather station
3. Project or intervention dates

Pandas Data Formats
~~~~~~~~~~~~~~~~~~~

We use :any:`pandas:pandas` data formats in order to take advantage of the powerful data
analysis tools provided in that package that users may already be familiar with. The
specifics of these formats are discussed in more detail below.

Please refer directly to the excellent pandas documentation for instructions for loading
data (e.g., :any:`pandas.read_csv`). The eemeter does come packaged with loading
methods, but these will only work for particular data formats. Here are some useful
eemeter methods for loading and manipulating data:

- :any:`eemeter.meter_data_from_csv`: Load meter data from CSV.
- :any:`eemeter.temperature_data_from_csv`: Load temperature data from CSV.
- :any:`eemeter.meter_data_from_json`: Load meter data from JSON.
- :any:`eemeter.temperature_data_from_json`: Load temperature data from JSON.
- :any:`eemeter.samples`: Return a list of sample data names.
- :any:`eemeter.load_sample`: Load sample data by name.
- :any:`eemeter.as_freq`: Coerce meter data into a different frequency.

.. _meter-data:

Meter Data
~~~~~~~~~~

Meter data is stored as a :any:`pandas.DataFrame` with a :any:`pandas.DatetimeIndex`.
Your data must be in the format demonstrated below to work with the eemeter library.

By convention,

1. Meter data are stored by period *start date*.
2. The length of each period is determined by the start date of the next value (which
   may be null).
3. The end date of the last period is given by a single nan-valued period appended at
   the end of the data for completeness.
4. The name of the dataframe column is "value". Units must be tracked separately.
5. The datetimes in the index must be timezone-aware.

Some examples of the eemeter meter data format::

    import numpy as np
    import pandas as pd

    # one year of daily data
    meter_data = pd.DataFrame(
        {"value": [1] * 365 + [np.nan]},
        index=pd.date_range("2018-01-01", "2019-01-01", freq="D", tz="UTC", name="start")
    )

    # two years of monthly data
    meter_data = pd.DataFrame(
        {"value": [1] * 24 + [np.nan]},
        index=pd.date_range("2017-01-01", "2019-01-01", freq="MS", tz="UTC", name="start")
    )

    # three months of 15-minute interval data
    meter_data = pd.DataFrame(
        {"value": [1] * 90 * 24 * 4 + [np.nan]},
        index=pd.date_range("2018-01-01", "2018-04-01", freq="15T", tz="UTC", name="start")
    )

.. _temperature-data:

Temperature Data
~~~~~~~~~~~~~~~~

Temperature data is stored as a :any:`pandas.Series` with a :any:`pandas.DatetimeIndex`.
While temperature data from any source can be used, the eeweather library is designed
specificially to provide temperature data from public sources for eemeter users.

 :any:`eeweather:index`

The eeweather library helps perform site to weather station matching and can pull
temperature data directly from public (US) data sources.

By convention,

1. Temperature data must be given with an hourly frequency.
2. The datetimes in the index must be timezone-aware.

An example of the eemeter temperature data format::

    import numpy as np
    import pandas as pd

    # three months of hourly interval data
    temperature_data = pd.Series(
        [1] * 24 * 90 + [np.nan],
        index=pd.date_range('2018-01-01', '2018-04-01', freq='H', tz='UTC')
    )

Using EEweather
~~~~~~~~~~~~~~~

Given a site location specified by lat/long coordinate, eeweather can find an
appropriate nearby weather station within the same climate zone and pull temperature
data directly from public sources::

    # requires user to `$ pip install eeweather sqlalchemy`
    from datetime import datetime
    import pytz
    import eeweather

    latitude = 38.1
    longitude = -118.3
    ranked_stations_closest_within_climate_zone = eeweather.rank_stations(
        latitude,
        longitude,
        match_iecc_climate_zone=True,
        match_iecc_moisture_regime=True,
        match_ba_climate_zone=True,
        match_ca_climate_zone=True,
        max_distance_meters=100000,
    )

    ranked_stations_closest_anywhere = eeweather.rank_stations(
        latitude,
        longitude,
    )

    ranked_stations = eeweather.combine_ranked_stations([
        ranked_stations_closest_within_climate_zone,
        ranked_stations_closest_anywhere,
    ])

    start_date = datetime(2018, 1, 1, tzinfo=pytz.UTC)
    end_date = datetime(2019, 1, 1, tzinfo=pytz.UTC)
    selected_station, warnings = eeweather.select_station(
        ranked_stations,
        coverage_range=(start_date, end_date)
    )
    selected_station.usaf_id
    temp_degC, warnings = selected_station.load_isd_hourly_temp_data(
        start_date, end_date
    )
    temp_degF = temp_degC * 9 / 5 + 32


.. _sample-data:

Sample Data
~~~~~~~~~~~

If you'd like to continue with this tutorial without loading in your own data,
you can use the fake data provided as samples along with this library::

    # hourly
    meter_data, temperature_data, sample_metadata = (
        eemeter.load_sample("il-electricity-cdd-hdd-hourly")
    )

    # daily
    meter_data, temperature_data, sample_metadata = (
        eemeter.load_sample("il-electricity-cdd-hdd-daily")
    )

    # other samples
    sample_names = eemeter.samples()

.. _baseline-model:

Building a Baseline Model
-------------------------

The CalTRACK methods require building a model of the usage during the baseline period
and then projecting that forward into the reporting period to calculate avoided energy
use. Before we can build the baseline model we need to get isolate 365 days of meter
data immediately prior to the end of the baseline period.

This method pulls data for a 365 baseline period by slicing backward from a project
date::

    import pandas as pd
    import eemeter
    from dateime import datetime
    import pytz

    meter_data = pd.DataFrame(
        {"value": [1] * 730 + [np.nan]},
        index=pd.date_range("2017-01-01", "2019-01-01", freq="D", tz="UTC", name="start")
    )
    baseline_end_date = datetime(2018, 6, 1, tzinfo=pytz.UTC)
    baseline_meter_data, warnings = eemeter.get_baseline_data(
        meter_data, end=baseline_end_date, max_days=365
    )

- :any:`eemeter.get_baseline_data`: Filter a dataset to baseline period data.

With baseline data isolated, we can build a baseline model. There are currently
two options for this: hourly and billing/daily.

CalTRACK Daily/Billing Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The CalTRACK daily and billing methods specifiy a way of modeling the weather-dependent
energy signature of a building. It selects a model which fits the data as well as
possible from a selection of candidate models. The parameters of the model
are heating and cooling balance points (i.e., the temperatures at which heating/cooling
related energy use tend to kick in), and the heating and cooling beta parameters, which
define the slope of the energy response to incremental differences between outdoor
temperature and the balance point. We'll do a grid search over possible heating and
cooling balance points and fit models to the `heating and cooling degree days
<https://en.wikipedia.org/wiki/Heating_degree_day>`_ defined by the outdoor temperatures
and each of those balance points. To do this, we precompute the heating and cooling
degree days using the methods below before we feed them into the modeling routines.

- :any:`eemeter.create_caltrack_daily_design_matrix`: Create a design matrix for
  CalTRACK daily methods.
- :any:`eemeter.create_caltrack_billing_design_matrix`: Create a design matrix for
  CalTRACK billing methods.
- :any:`eemeter.compute_usage_per_day_feature`: Transform meter data into usage per day.
- :any:`eemeter.compute_temperature_features`: Compute heating and cooling degree days
  and other useful temperature features.
- :any:`eemeter.merge_features`: Combine a list of Dataframe or Series objects which
  share an index into a single DataFrame.
- :any:`caltrack-compliance`: Steps to ensure full CalTRACK compliance.

To run the CalTRACK Daily/Billing methods::

    # create a design matrix suitable for use with daily data
    baseline_design_matrix = eemeter.create_caltrack_daily_design_matrix(
        baseline_meter_data, temperature_data,
    )

    # create a design matrix suitable for use with billing data
    baseline_design_matrix = eemeter.create_caltrack_billing_design_matrix(
        baseline_meter_data, temperature_data,
    )

    # build a CalTRACK model
    baseline_model = eemeter.fit_caltrack_usage_per_day_model(
        baseline_design_matrix,
    )

These methods are shortcuts. Behind the scenes, they combine meter data and temperature
data into a single DataFrame using :any:`eemeter.compute_usage_per_day_feature` to
transform the meter data into usage per day and
:any:`eemeter.compute_temperature_features` to create a a search grid of heating and
cooling degree day values. The shortcut methods use this case, we'll use the wide
balance point ranges recommended by CalTRACK.  The shortcut method can combines the two
using :any:`eemeter.merge_features`.

If using billing data, note that the values represented in the design matrix created by
calling :any:`eemeter.compute_usage_per_day_feature` are returned as average usage per
day, as specified by the CalTRACK methods, not as totals per period, as they are
represented in the inputs. The heating/cooling degree days returned by
:any:`compute_temperature_features` are also average heating/cooling degree days per
day, and not total heating/cooling degree days per period. This averaging behavior can
be modified with the `use_mean_daily_values` parameter, which is set to `True` by
default.

CalTRACK Hourly Methods
~~~~~~~~~~~~~~~~~~~~~~~

The CalTRACK hourly methods require a multi-stage dataset creation process which is a
bit more involved than the daily/billing dataset creation process above. There are two
primary reasons for this extra complexity. First, unlike the daily/billing methods, the
hourly methods build separate models for each calendar month, which adds a few extra
steps. Second, also unlike the billing and daily methods, there are two features of the
dataset creation which must themselves be fitted to a preliminary dataset occupancy
features and temperature bin features.

**Preliminary design matrix**

The preliminary design matrix has some simple time and temperature features. These features do
not vary by segment and are precursors to other features (see below for a better
explanation of segmentation). This step looks a lot like the daily/billing dataset
creation. These features are used subsequently to fit the occupancy and temperature bin
features.

- :any:`eemeter.create_caltrack_hourly_preliminary_design_matrix`: Create a design
  matrix for the first stage of CalTRACK hourly.
- :any:`eemeter.compute_time_features`: Create a time feature for the index
  (`time_of_week`).
- :any:`eemeter.compute_temperature_features`: Compute heating and cooling degree days
  and other useful temperature features.
- :any:`eemeter.merge_features`: Combine a list of Dataframe or Series objects which
  share an index into a single DataFrame.

The preliminary design matrix has only two fixed heating (50 degF) and cooling (65 degF)
degree day columns - these are used to fit the occupancy model. It also has an hour
of week column, which is a categorical variable indicating the hour of the week using an
integer from 0 to 167 (i.e., 7 days * 24 hours/day). `0` is Monday midnight to 1am.

**Segmentation**

CalTRACK hourly requires creating independent models for each month of a dataset. The
eemeter package calls this "segmentation". Segmentation breaks a dataset into $n$ named
and weighted subsets.

Before we can move on to the next steps of creating the CalTRACK hourly dataset, we need
to create a monthly segmentation for the hourly data. We will use this to create 12
independent hourly models - one for each month of the calendar year. The eemeter
function for creating these weights is called :any:`eemeter.segment_time_series` and it
takes a :any:`pandas.DatetimeIndex` as input.

This segmentation matrix contains 1 column for each segment (12 in all), each of which
contains the segmentation weights for that column. The segmentation scheme we use here
is to have one segment for each month which contains a single fully weighted calendar
month and two half-weighted neighboring calendar months. The eemeter code name for this
segmentation scheme is called `'three_month_weighted'` (There's also `'all'`, `'one_month'`,
and `'three_month'`, each of which behaves a bit differently).

We are creating this segmentation over the time index of the baseline period that is
represented in the preliminary hourly design matrix.

- :any:`eemeter.segment_time_series`: Create a segmentation using the specified scheme.


**Occupancy**

Occupancy is estimated by building a simple model from the preliminary design matrix
hdd_50 and cdd_65 columns. This is done for each segment independently, so results are
returned as a dataframe with one segment of results per column. The `segmentation`
argument indicates that the analysis should be done once per segment. Occupancy is
determined by hour of week category. A value of 1 for a particular hour indicates an
"occupied" mode, and a value of 0 indicates "unoccupied" mode. These modes are
determined by the tendency of the hdd_50/cdd_65 model to over- or under-predict usage
for that hour, given a particular threshold between 0 and 1 (default 0.65) (if the
percent of underpredictions (by count) is lower than that threshold, then the mode is
"unoccupied", otherwise the mode is "occupied").

The occupancy lookup is organized by hour of week (rows) and model segment (columns).

- :any:`eemeter.estimate_hour_of_week_occupancy`: Estimate occupancy by time of week for
  each segment.

**Fitting segmented temperature bins**

Temperature bins are fit for each segment such that each bin has sufficient number of
temperature readings (20 per bin, by default). Bins are defined by starting with a
proposed set of bins (see the `default_bins` argument) and systematically dropping bin
endpoints if they do not meet sufficiency requirements. Bins themselves are not dropped
but are effectively combined with neighboring bins. Except for the fact that
zero-weighted times are dropped, segment weights are not considered when fitting
temperature bins.

Because bin fitting and validation is done independently for each segment, results are
returned as a dataframe with one segment of results per column. The contents of the
dataframe are boolean indicators of whether the bin endpoint should be used for
temperatures in that segment. Some bin endpoints are dropped because of insufficient
reading counts. The bin endpoints that are dropped for each segment are given a value of
`False`. You'll notice in this dataset that the the winter months tend to have combined
high temperature bins and the summer months tend to have combined low temperature bins.

- :any:`eemeter.fit_temperature_bins`: Fit temperature bins to data, dropping bin
  endpoints for bins that do not meet the minimum temperature count such that remaining
  bins meet the minimum count.

With these features in hand, now we can combine them into a segmented dataset using the
helper function :any:`eemeter.iterate_segmented_dataset` and a prefabricated feature
processor :any:`eemeter.caltrack_hourly_fit_feature_processor` which is provided to assist
creating the segmented dataset given a preliminary design matrix of the form created
above. The feature processor transforms the each segment of the dataset using the
occupancy lookup and temperature bins created above. We are creating a python `dict` of
`pandas.Dataframes` - one for each time series segment encountered in the baseline data.
The keys of the dict are segment names. The values are DataFrame objects containing the
exact data needed to fit the a CalTRACK hourly model.

**Putting it all together**

- :any:`eemeter.fit_caltrack_hourly_model`: Build a model which combines models for each
  segment.

To run the CalTRACK Hourly methods::

    # create a design matrix for occupancy and segmentation
    preliminary_design_matrix = (
        eemeter.create_caltrack_hourly_preliminary_design_matrix(
            baseline_meter_data, temperature_data,
        )
    )

    # build 12 monthly models - each step from now on operates on each segment
    segmentation = eemeter.segment_time_series(
        preliminary_design_matrix.index,
        'three_month_weighted'
    )

    # assign an occupancy status to each hour of the week (0-167)
    occupancy_lookup = eemeter.estimate_hour_of_week_occupancy(
        preliminary_design_matrix,
        segmentation=segmentation,
    )

    # assign temperatures to bins
    occupied_temperature_bins, unoccupied_temperature_bins = eemeter.fit_temperature_bins(
        preliminary_design_matrix,
        segmentation=segmentation,
        occupancy_lookup=occupancy_lookup,
    )

    # build a design matrix for each monthly segment
    segmented_design_matrices = (
        eemeter.create_caltrack_hourly_segmented_design_matrices(
            preliminary_design_matrix,
            segmentation,
            occupancy_lookup,
            occupied_temperature_bins,
            unoccupied_temperature_bins,
        )
    )

    # build a CalTRACK hourly model
    baseline_model = eemeter.fit_caltrack_hourly_model(
        segmented_design_matrices,
        occupancy_lookup,
        occupied_temperature_bins,
        unoccupied_temperature_bins,
    )

Computing CalTRACK metered savings
----------------------------------

Suppose we wanted to calculated metered savings for the year following a project
intervention. This could be accomplished by first slicing the original meter data down
to the subset in the first year following an intervention.

The :any:`eemeter.metered_savings` method performs the logic of estimating
counterfactual baseline reporting period usage. For this, it requires the fitted
baseline model, the reporting period meter data (for its index - so that it can be
properly joined later), and corresponding temperature data. Note that this method can
return results disaggregated into base load, cooling load, or heating load or as the
aggregated usage. We do both here for demonstration purposes.::

    reporting_meter_data, warnings = eemeter.get_reporting_data(
        meter_data, start=blackout_end_date, max_days=365
    )

    metered_savings, error_bands = eemeter.metered_savings(
        baseline_model, reporting_meter_data, temperature_data
    )

    # with disaggregated usage predictions, billing/daily only
    metered_savings, error_bands = eemeter.metered_savings(
        baseline_model, reporting_meter_data, temperature_data, with_disaggregated=True
    )

This method also returns error bands which can be used for calculating Fractional
Savings Uncertainty.

Computing (non-CalTRACK) typical year savings
---------------------------------------------

If we want to compute annual weather normalized modeled savings, we'll need a model of
reporting period usage in addition to the model of baseline period usage created in
the tutorial above.::

    import pandas as pd
    import eeweather

    # temperature data
    normal_year_temperatures = (
        eeweather.ISDStation('722880').load_tmy3_hourly_temp_data()
    )
    # dates over which to predict
    prediction_index = pd.date_range('2015-01-01', periods=365, freq='D', tz='UTC')
    annualized_savings = eemeter.modeled_savings(
        baseline_model, reporting_model,
        prediction_index, normal_year_temperatures, with_disaggregated=True
    )

Visualization
-------------

Fitted Billing/Daily models can be inspected by plotting an energy signature chart::

    ax = eemeter.plot_energy_signature(meter_data, temperature_data)
    baseline_model.plot(
        ax=ax, candidate_alpha=0.02, with_candidates=True, temp_range=(-5, 88)
    )

.. _cautions:

Cautions
--------

At time of writing (Sept 2019), the OpenEEmeter, as implemented in the ``eemeter``
package, contains the most complete open source implementation of the `CalTRACK methods
<http://www.caltrack.org/>`_, which specify a way of calculating **avoided energy use**
at a single energy meter at a single site. However, using the OpenEEmeter to calculate
avoided energy use does not in itself guarantee compliance with the CalTRACK method
specification, nor is using the OpenEEmeter a requirement of the CalTRACK methods. The
eemeter package is a toolkit that may help with implementing a CalTRACK compliant
analysis, as it provides a particular implementation of the CalTRACK methods which
consists of a set of functions, parameters, and classes which can be configured to run
the CalTRACK methods and variants. Please keep in mind while using the package that the
eemeter assumes certain data cleaning tasks that are specified in the CalTRACK methods
have occurred *prior* to usage with the eemeter. The package will create warnings to
expose issues of this nature where possible.

The eemeter package is built for flexibility and modularity. While this is generally
helpful and makes it possible to do more with the package, one potential consequence of
this for users is that without being careful to follow the both the eemeter
documentation *and* the guidance provided in the CalTRACK methods, it is very possible
to use the eemeter in a way that does not comply with the CalTRACK methods. For example,
while the CalTRACK methods set specific hard limits for the purpose of standardization
and consistency, the eemeter can be configured to edit or entirely ignore those limits.
The main reason for this flexibility is that the emeter package is used not only to
comply with the CalTRACK methods, but also to develop, test, and propose potential
changes to those methods.

Rather than providing a single method that directly calculates avoided energy use from
the required inputs, the eemeter library provides a series of modular functions that can
be strung together in a variety of ways. The tutorial below describes common usage and
sequencing of these functions, especially when it might not otherwise be apparent from
the :ref:`api-docs`.

Some new users have assumed that the eemeter package constitutes an entire application
suitable for running metering analytics at scale. This is not necessarily the case. It
is designed instead to be embedded within other applications or to be used in one-off
analyses. The eemeter is a toolbox that leaves to the user decisions about when to use
or how to embed the provided tools within other applications. This limitation is an
important consequence of the decision to make the methods and implementation as open and
accessible as possible.

As you dive in, remember that this is a work in progress and that we welcome feedback
and contributions. To contribute, please open an
`issue <https://github.com/openeemeter/eemeter/issues>`_ or a `pull
request <https://github.com/openeemeter/eemeter/pulls>`_ on github.
