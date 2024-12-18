Changelog
=========

Development
-----------

* Fix bug where HourlyBaselineData and HourlyReportingData wasn't actually NaNning zero rows when `is_electricity=True`.

4.0.7
-----

* Handle ambiguous and nonexistent local times when creating daily dataclass

4.0.6
-----


* Update docs.
* Update typehints on core daily and utility functions.
* Minor change to loading test data to ensure the reporting period is a year ahead of the baseline period.

4.0.5
-----


* Flip slope when deserializing legacy hdd_only models

4.0.4
-----

* Add support for deserializing legacy hourly models
* Fix legacy daily model deserialization

4.0.3
-----

* Move masking behavior for rows with missing temperature from reporting dataclass to prediction output
* Add disqualification check to billing model predict()

4.0.2
-----

* Force index to use nanosecond precision
* Compute coverage using same offset as initial reads to fix issues when downsampling hourly data
* Update test data location
* Fix bug in daily plotting to remove NaN values if input
* Refactor sufficiency criteria to be more explicit and easier to manage

4.0.1
-----

* Correct dataframe input behavior and final row temperature aggregation
* Remove unnecessary datetime normalization in order to respect hour of day
* Convert timestamps in certain warnings to strings to allow serialization
* Allow configuration of segment_type in HourlyModel wrapper


4.0.0
-----

* Update daily model methods, API, and serialization
* Provide new API for hourly model to match daily syntax and prepare for future additions
* Add baseline and reporting dataclasses to support compliant initialization of meter and temperature data

3.2.0
-----

* Addition of modules and amendments in support of international facility for EEMeter, including principally:
* Addition of quickstart.py; updating setup.py and __init__/py accordingly.
* Inclusion of temperature conversion amendments to design_matrices; features; and derivatives.
* Addition of new tests and samples.
* Amendments to tutorial.ipynb.
* Addition of eemeter international.ipynb.
* Change .iteritems() to .items() in accordance with pandas>=2.0.0
* .get_loc(x, method=...) to .get_indexer([x],method=...)[0] in accordance with pandas>=2.0.0
* Updated mean() to mean(numeric_only=True) in accordance to pandas>=2.0.0
* Updated tests to work with pandas>=2.0.0
* Update python version in Dockerfile.
* Update other dependencies (including adding rust) in Dockerfile.
* Remove pinned dependencies in Pipfile.
* Relock Pipfile (and do so inside of the docker image).
* Update pytests to account for changes in newer pandas where categorical variables are no longer included in `df.sum().sum()`.
* Clarify the functioning of start, end and max_days parameters to `get_reporting_data()` and `get_baseline_data()`.

3.1.1
-----

* Update observed_mean calculation to account for solar (negative usage) to provide
sensible cvrmse calculations.

3.1.0
-----

* Remove missing hour_of_week categories in the CalTrack hourly methods so they predict null for those hours. 

3.0.0
-----

* Remove python27 support.
* Update Pipfile lock.
* Update `fit_temperature_bins` to potentially take an `occupancy_lookup` in order to
  fit different temperature bins for occupied/unoccupied modes. *This changes the args passed to eemeter.create_caltrack_hourly_segmented_design_matrices, where it now requires a set of bins for occupied and unoccupied temperatures separately.*
* Update CalTRACK hourly model formula to use different bins for occupied and
  unoccupied mode.

2.10.11
-------

* Fix tests and make changes to ensure tests pass on pandas version 1.2.1.
* Fix bug in segmentation.py causing a section of tutorial to fail.

2.10.0
------

* Add additional terms into ModelMetrics() class which can be used in fractional savings uncertainy computations.

2.9.2
-----

* Remove fixing of versions of libraries in setup.py to avoid unforeseen issues with library updates.

2.9.1
-----

* Fix versions of libraries in setup.py to avoid unforeseen issues with library updates.

2.9.0
-----

* Clarify blackout period.

2.8.6
-----

* Fix issue with `get_reporting_data` and `get_baseline_data` when passing data with non-UTC timezones.

2.8.5
-----

* Add functions to clean billing/daily data according to caltrack rules.

2.8.4
-----

* Further limit segments used in hourly `totals_metrics` to only calculate when weight=1.

2.8.3
-----

* Update hourly `totals_metrics` calculation to properly use only the segment of the model.

2.8.2
-----

* Add `totals_metrics` to hourly models.

2.8.1
-----

* Fix bug with `get_baseline_data` in regards to recent addition of `n_days_billing_period_overshoot` kwarg.

2.8.0
-----

* Update `get_baseline_data` to allow for limit to billing overshoot using `n_days_billing_period_overshoot` kwarg.

2.7.7
-----

* Add function to clean billing data to fit caltrack specifications (`clean_caltrack_billing_data`).

2.7.6
-----

* Update io functions to support latest pandas (>=0.24.x).
* Update documentation for CalTRACK Hourly methods.
* Add tutorial.

2.7.5
-----

* Fix completeness check for `get_terms` for last term.

2.7.4
-----

* Make more usable outputs for the `get_terms` function (list of eemeter.Term objects).

2.7.3
-----

* Update `as_freq` so it has an optional `include_coverage` parameter where it returns a dataframe with one column including the percent coverage of data used to create each sample.

2.7.2
-----

* Fixes the columns that are given in an empty prediction result called with the
  ` with_design_matrix=True` flag set for caltrack usage per day methods.
* Update bug report github issue template.
* Add test for `as_freq`.

2.7.1
-----

* Change `as_freq` to handle all Null series.

2.7.0
-----

* Add `get_terms` method to allow splitting reporting data into any number
  of terms specified by day length.

2.6.0
-----

* Change `fit_caltrack_hourly_model` so it returns a `CalTRACKHourlyModelResults` object rather than a `CalTRACKHourlyModel`, in order to bring it in line with the `caltrack_usage_per_day` model outputs.

2.5.4-post1
-----------

* Update MANIFEST.in to fix release and update `./bump_version.sh` script
  to remove build directories.

2.5.4
-----

* Add data fields to the `DataSufficiency` even if there are no warnings when calculating sufficiency.

2.5.3-post2
-----------

* Attempt 2 to fix release .whl file by removing local build and dist
  directories before running `python setup.py upload`.

2.5.3-post1
-----------

* Fix release .whl file which had some extra directories.
* Add draft MAINTAINERS.md.

2.5.3
-----

* Fix `metered_savings` behavior so that it does not fail to compute error bands when there is 0 variance in the baseline.

2.5.2
-----

* Fix `as_freq` behavior to preserve sum and add a null last index at the target
  frequency if necessary.

2.5.1
-----

* Capture an additional exception type (`KeyError`) in recently adjusted
  `get_baseline_data` and `get_reporting_data` methods.

2.5.0
-----

* Add parameters to `get_baseline_data` and `get_reporting_data` to help make
  these methods a bit more correct for billing data.
* Preserve nulls properly in `as_freq`.
* Update jupyter version to be compatible with latest tornado version.

2.4.0
-----

* Fix for bug that occasionally leads to `LinAlgError: SVD did not converge` error when fitting caltrack hourly models by addressing multi-collinearity when only a single occupancy mode is detected

2.3.1
-----

* Hot fix for bug that occasionally leads to `LinAlgError: SVD did not converge` error when fitting caltrack hourly models by converting the weights from `np.float64` ton `np.float32`.

2.3.0
-----

* Fix bug where the model prediction includes features in the last row that should be null.
* Fix in `transform.get_baseline_data` and `transform.get_reporting_data` to enable pulling a full year of data even with irregular billing periods

2.2.10
------

* Added option in `transform.as_freq` to handle instantaneous data such as temperature and other weather variables.

2.2.9
-----

* Predict with empty formula now returns NaNs.

2.2.8
-----

* Update `compute_occupancy_feature` so it can handle instances where there are less than 168 values in the data.

2.2.7
-----

* SegmentModel becomes CalTRACKSegmentModel, which includes a hard-coded check that the same hours of week are in the model fit parameters and the prediction design matrix.

2.2.6
-----

* Reverts small data bug fix.

2.2.5
-----

* Fix bug with small data (1<week) for hourly occupancy feature calculation.
* Bump dev eeweather version.
* Add `bump_version` script.
* Filter two specific warnings when running tests:
  statsmodels pandas .ix warning, and eemeter model fitting warning.

2.2.4
-----

* Add `json()` serialization for `SegmentModel` and `SegmentedModel`.

2.2.3
-----

* Change `max_value` to float so that it can be json serialized even if the input is int64s.

2.2.2
-----

* Add warning to `caltrack_sufficiency_criteria` regarding extreme values.

2.2.1
-----

* Fix bug in fractional savings uncertainty calculations using billing data.

2.2.0
-----

* Add fractional savings uncertainty to modeled savings derivatives.

2.1.8
-----

* Update so that models built with empty temperature data won't result in error.

2.1.7
-----

* Update so that models built from a single record won't result in error.

2.1.6
-----

* Update multiple places where `df.empty` is used and replaced with `df.dropna().empty`.
* Update documentation for running CalTRACK hourly methods.

2.1.5
-----

* Fix zero division error in metrics calculation for several metrics that
  would otherwise cause division by zero errors in fsu_error_band calculation.

2.1.4
-----

* Fix zero division error in metrics calculation for series of length 1.

2.1.3
-----

* Fix bug related to caltrack billing design matrix creation during empty temperature traces.

2.1.2
-----

* Add automatic t-stat computation for metered savings error bands, the
  implementation of which requires expicitly adding scipy to setup.py
  requirements.
* Don't compute error bands if reporting period data is empty for metered
  savings.

2.1.1
-----

* Fix degree day ranges (30-90) for prefab caltrack design matrix creation
  methods.
* Fix the warning for total degree days to use total degree days instead of
  average degree days.

2.1.0
-----

* Update the `use_billing_presets` option in `fit_caltrack_usage_per_day_model`
  to use a minimum data sufficiency requirement for qualifying CandidateModels
  (similar to daily methods).
* Add an error when attempting to use billing presets without passing a weights
  column to facilitate weighted least squares.

2.0.5
-----

* Give better error for duplicated meter index in compute temperature features.

2.0.4
-----

* Change metrics input length error to warning.

2.0.3
-----

* Apply black code style for easy opinionated PEP 008 formatting
* Apply JSON-safe float conversion to all metrics.

2.0.2
-----

* Cont. fixing JSON representation of NaN values

2.0.1
-----

* Fixed JSON representation of model classes

2.0.0
-----

* Initial release of 2.x.x series
