CalTRACK Compliance
===================

.. role:: red

Checklist for caltrack compliance:


Section 2.2: Data Constraints
-----------------------------


Section 2.2.1: Missing Values/Data Sufficiency
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **2.2.1.1**: :any:`eemeter.get_baseline_data` must set ``max_days=365``.
- **2.2.1.2**: :any:`eemeter.caltrack_sufficiency_criteria` must set ``min_fraction_daily_coverage=0.9``
- **2.2.1.3**: (Data Preparation) Missing values in input data have been represented as ``float('nan')``, ``np.nan``, or anything recognized as null by the method :any:`pandas.isnull`.
- **2.2.1.4**: (Data Preparation) Values of ``0`` in electricity data have been converted to ``np.nan``.


Section 2.2.2: Daily Data Sufficiency
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **2.2.2.1**: (Data Preparation) Input meter data has been appropriately downsampled to daily values.
- **2.2.2.2**: (Data Preparation) Estimated reads in input data have been combined with subsequent reads.
- **2.2.2.3**: :any:`eemeter.merge_temperature_data` sets ``percent_hourly_coverage_per_day=0.5``.


Section 2.2.3: Billing Data Sufficiency
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **2.2.3.1**: (Data Preparation) Estimated reads in input data have been combined with subsequent reads up to a 70 day limit. Estimated reads count as missing values when evaluating the sufficiency criteria defined in 2.2.1.2. 
- **2.2.3.2**: :any:`eemeter.merge_temperature_data` must set ``percent_hourly_coverage_per_billing_period=0.9`` and `:any:`eemeter.caltrack_sufficiency_criteria` must set ``min_fraction_daily_coverage=0.9``.
- **2.2.3.3**: (Data Preparation) Input meter data that represents billing periods less than 25 days long has been converted to ``np.nan``.
- **2.2.3.4**: (Data Preparation) Input meter data that represents billing periods greater than 35 days long for pseudo-monthly billing period calculations and 70 days long for bi-monthly billing period calculations has been converted to ``np.nan``.


Section 2.2.X: Other Data Sufficiency Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **2.2.4**: :any:`eemeter.caltrack_sufficiency_criteria` set `requested_start_date` and `requested_end_date` to receive critical warnings related to data outside of the requested period of analysis.
- **2.2.5**: (Data Preparation) Projects have been removed if the status of net metering has changed during the baseline or reporting periods.
- **2.2.6**: (Data Preparation) Projects have been removed if EV charging has been installed during the baseline or reporting periods.



Section 2.3: Data Quality
-------------------------


Section 2.3.1: Impossible Dates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **2.3.1.1**: (Data Preparation) For billing analysis, input meter data containing invalid dates for a valid month have been converted to the first date of that month.
- **2.3.1.2**: (Data Preparation) Input meter data containing invalid months/years for have been removed and a warning has been generated.


Section 2.3.2: Duplicate Records
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


- **2.3.2.1**: (Data Preparation) Meter usage and temperature data has used matching time zone information to ensure that the upsampled values represent the same periods of time.
- **2.3.2.2**: *Not yet compliant (If duplicate rows are found for meter data, then the project must be flagged as it may have sub-metering/multiple meters. Warnings could possibly be generated in :any:`eemeter.remove_duplicates`.*


Section 2.3.X: Other Data Quality Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


- **2.3.3**: :any: `eemeter.merge_temperature_data` ``meter_data`` and ``temperature_data`` must be timezone-aware and have matching timezones. 
- **2.3.4**: If NOAA weather data was used (which is roughly hourly), it has been normalized to hourly using :any:`eeweather.ISDStation.fetch_isd_hourly_temp_data`.
- **2.3.5**: Warnings are generated in :any:`eemeter.caltrack_sufficiency_criteria` if negative meter values are discovered as they indicate the possible presence of unreported net metering.
- **2.3.6**: (Data Preparation) Must generate warning for values that are more than three interquartile ranges larger than the median usage.
- **2.3.7**: (Audit) Resulting dataset of meter runs has been compared with expected counts of sites, meters, and projects.
- **2.3.8**: (Data Preparation) Meter data has been downsampled according to the desired frequency for analysis using :any:`eemeter.as_freq` before merging of temperature data or modeling.


Section 2.4: Matching Sites to Weather Stations
-----------------------------------------------

- **2.4.1**: When matching weather stations to sites, :any:`eeweather.match_lat_long` should use the default ``mapping`` parameter.
- **2.4.2**: When matching a particular site to a weather station, a weather station mapping :any:`eeweather.ISDStationMapping` generates a warning if the weather station is greater than 200 km from the site.


Section 3.2: Balance Points
---------------------------

- **3.2.1**: When calculating cooling and heating degree days :any:`eemeter.merge_temperature_data` must set ``heating_balance_points`` to be any list ranging from 30 to 90 with a maximum gap of 3 degrees Fahrenheit. For electricity meter use data, ``cooling_balance_points`` must also be any list ranging from 30 to 90 with a maximum gap of 3 degrees Fahrenheit. For natural gas meter use data, the function must set `fit_cdd=False` and ``cooling_balance_points=None`` so that models using cooling degree days are not considered.
- **3.2.2.1**: :any:`eemeter.get_cdd_hdd_candidate_models` only generates cdd_hdd candidate models where the cooling balance point is greater than or equal to the heating balance point.
- **3.2.2.2**: For daily data, :any:`eemeter.caltrack_method` must set ``minimum_non_zero_cdd=10, minimum_non_zero_hdd=10, minimum_total_cdd=20, minimum_total_hdd=20``. For billing data, :any:`eemeter.caltrack_method` must set ``use_billing_presets=True``.
- **3.2.3**: See the description above regarding the gap between candidate balance points in **3.2.1**. 


Section 3.3: Design Matrix (for Daily and Billing Methods)
----------------------------------------------------------

- **3.3.1**: :any:`eemeter.caltrack_method` is used for model candidate creation and model selection. It uses one of the following functions to construct models with the formula ``meter_value ~ hdd_X + cdd_Y``, where X is the heating balance point and Y is the cooling balance point. This is specifically done in one of the following functions: :any:`eemeter.get_single_cdd_only_candidate_model`, :any:`eemeter.get_single_hdd_only_candidate_model`, :any:`eemeter.get_single_cdd_hdd_candidate_model`. 
- **3.3.1.1**: For billing methods, :any:`eemeter.merge_temperature_data` must set ``use_mean_daily_values=True``.
- **3.3.1.2**: :any:`eemeter.merge_temperature_data` must set ``degree_day_method='daily'``.  
- **3.3.1.3**: The output of :any:`eemeter.caltrack_method` is a :any:`eemeter.ModelResults`. If a model has been selected, then :any:`eemeter.ModelResults` contains an attribute ``model`` which is a :any:`CandidateModel`. This :any:`CandidateModel` contains an attribute ``model_params`` which is a dictionary containing model parameters. It potentially can contain the following parameters: ``intercept``, ``beta_cdd``, ``cooling_balance_point``, ``beta_hdd``, and ``heating_balance_point`` depending on whether ``this_model_results.model_type`` is ``intercept_only``, ``cdd_only``, ``hdd_only``, or ``cdd_hdd``.


Section 3.4: Fit Candidate Models
---------------------------------

- **3.4.1**: For daily methods, :any:`eemeter.caltrack_method` must set ``weight_cols=None``.
- **3.4.2**: For billing methods, :any:`eemeter.caltrack_method` must set ``weight_cols='n_days_kept'``. 
- **3.4.3.1**: :any:`eemeter.caltrack_method` must set ``fit_cdd=True, fit_intercept_only=True, fit_cdd_only=True, fit_hdd_only=True, fit_cdd_hdd=True`` for electricity data, and ``fit_cdd=False, fit_intercept_only=True, fit_cdd_only=False, fit_hdd_only=True, fit_cdd_hdd=False`` for gas data.  
- **3.4.3.2**: :any:`eemeter.caltrack_method` calls the following functions to generate candidate models (given that the correct parameters are set to true as defined in **3.4.3.1**: :any:`eemeter.get_single_cdd_only_candidate_model`, :any:`eemeter.get_single_hdd_only_candidate_model`, :any:`eemeter.get_single_cdd_hdd_candidate_model`, :any:`eemeter.get_intercept_only_candidate_models`. Within each of these functions, the status of the model is set to 'DISQUALIFIED' and a warning is generated if any model parameters are negative. 
- **3.4.3.3**: :any:`eemeter.caltrack_method` calls  :any:`eemeter.select_best_candidate` to select the best of the model candidates. This function finds the best of the model candidates based on which model has the highest adjusted r-squared value. 


Section 3.5: Computing Derived Quantities 
-----------------------------------------

- **3.5.1**: :any:`eemeter.caltrack_metered_savings` returns a :any:`pandas.DataFrame` with a column ``metered_savings`` which contains the avoided energy use values.
