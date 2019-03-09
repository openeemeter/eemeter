Changelog
=========

Development
-----------

* Placeholder

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
