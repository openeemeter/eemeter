Changelog
=========

Development
-----------

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
