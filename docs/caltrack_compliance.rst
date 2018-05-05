CalTRACK Compliance
===================

.. role:: red

Checklist for caltrack compliance:


Section 2.2: Data Constraints
-----------------------------


Section 2.2.1: Missing Values/Data Sufficiency
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **2.2.1.1**: :any:`eemeter.get_baseline_data` must set ``max_days=365``.
- **2.2.1.2**: :any:`eemeter.caltrack_sufficiency_criteria` must set ``min_fraction_daily_coverage=0.9``.
- **2.2.1.3**: (ETL) Missing values in input data have been represented as ``float('nan')``, ``np.nan``, or anything recognized as null by the method :any:`pandas.isnull`.
- **2.2.1.4**: (ETL) Values of ``0`` in electricity data have been converted to ``np.nan``.


Section 2.2.2: Daily Data Sufficiency
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **2.2.2.1**: (ETL) Input meter data has been appropriately downsampled to daily values.
- **2.2.2.2**: (ETL) Estimated reads in input data have been combined with subsequent reads.
- **2.2.2.3**: :any:`eemeter.merge_temperature_data` sets ``percent_hourly_coverage_per_day=0.5``.
- **2.2.2.4**: (ETL) Meter usage and temperature data that is downsampled to daily has used matching time zone information to ensure that the upsampled values represent the same periods of time.


Section 2.2.3: Billing Data Sufficiency
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **2.2.3.1**: (ETL) Estimated reads in input data have been combined with subsequent reads up to a 70 day limit. Estimated reads count as missing values when evaluating the sufficiency criteria defined in 2.2.1.2. 
- **2.2.3.2**: :any:`eemeter.caltrack_sufficiency_criteria` must set ``min_fraction_daily_coverage=0.9``.
- **2.2.3.3**: (ETL) Input meter data that represents billing periods less than 25 days long has been converted to ``np.nan``.
- **2.2.3.4**: (ETL) Input meter data that represents billing periods greater than 35 days long for pseudo-monthly billing period calculations and 70 days long for bi-monthly billing period calculations has been converted to ``np.nan``.


Section 2.2.X: Other Data Sufficiency Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **2.2.4**: :any:`eemeter.caltrack_sufficiency_criteria` set `requested_start_date` and `requested_end_date` to receive warnings related to data outside of the requested period of analysis.
- **2.2.5**: (ETL) Projects have been removed if the status of net metering has changed during the baseline or reporting periods.
- **2.2.6**: (ETL) Projects have been removed if EV charging has been installed during the baseline or reporting periods.



Section 2.3: Data Quality
-------------------------


Section 2.3.1: Impossible Dates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **2.3.1.1**: (ETL) Input meter data containing invalid dates for a valid month have been converted to the first date of that month.
- **2.3.1.2**: (ETL) Input meter data containing invalid months/years for have been removed and a warning has been generated.


Section 2.3.2: Duplicate Records
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


- **2.3.2.1**: (ETL) Meter usage and temperature data has used matching time zone information to ensure that the upsampled values represent the same periods of time.
- **2.3.2.2**: (ETL) Multiple sources of usage and temperature data have been combined into a single :any:`pandas.DataFrame` for meter data and a single :any:`pandas.DataFrame` for temperature data. Duplicate rows are removed using :any:`eemeter.remove_duplicates`.


Section 2.3.X: Other Data Quality Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **2.3.3**: If NOAA weather data was used (which is roughly hourly), it has been normalized to hourly using :any:`eeweather.ISDStation.fetch_isd_hourly_temp_data`.
- **2.3.4**: (ETL) If multiple project installation dates were given, the earliest date is assigned to be the blackout_start_date (when the baseline period ends) and :any:`eemeter.get_baseline_data` must set ``end=blackout_start_date``. The latest date is assigned to be the blackout_end_date (when the reporting period begins) and :any:`eemeter.get_reporting_data` must set ``start=blackout_end_date``.
- **2.3.5**: Warnings are generated in :any:`eemeter.caltrack_sufficiency_criteria` if negative meter values are discovered as they indicate the possible presence of unreported net metering.
- **2.3.6**: *Not yet compliant (Must generate warning for values that are more than three interquartile ranges larger than the median usage).*
- **2.3.7**: (Tests) Resulting dataset of meter runs has been compared with expected counts of sites, meters, and projects.
- **2.3.8**: Meter data is downsampled according to the desired frequency for analysis using :any:`eemeter.as_freq` before merging of temperature data or modelling.


Section 2.4: Matching Sites to Weather Stations
-----------------------------------------------

- **2.4.1**: When matching weather stations to sites, :any:`eeweather.match_lat_lng` and :any:`eeweather.match_zcta` must set ``mapping=None`` or ``mapping=mappings.oee_zcta``.
- **2.4.2**:*Not yet compliant (Must generate warning when station is >200 km from site).* 


Section 3.2: Balance Points
------------------------

- **3.2.1**: When calculating cooling and heating degree days :any:`eemeter.merge_temperature_data` must set ``heating_balance_points=range(30,90,X)``. For electricity meter use data, that function must set ``cooling_balance_points=range(30,90,X)``. For natural gas meter use data, that function must set ``cooling_balance_points=None``. X can be 1,2, or 3 (meaning that the gap between candidate balance points is less than or equal to 3 degrees).
- **3.2.2.1**: *Not yet compliant (Must generate DISQUALIFIED model status in :any:eemeter.get_single_cdd_hdd_candidate_model if cooling_balance_point < heating_balance_point.*
- **3.2.2.2**: :any:`eemeter.caltrack_method` must set ``minimum_non_zero_cdd=10, minimum_non_zero_hdd=10, minimum_total_cdd=20, minimum_total_hdd=20``
- **3.2.3**: See the description above regarding the gab between candidate balance points in **3.2.1**. 
