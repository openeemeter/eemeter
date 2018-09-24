Changelog
=========

Development
-----------

* Update the `use_billing_presets` option in `caltrack_method` to use a minimum data sufficiency requirement for qualifying CandidateModels (similar to daily methods).
* Add an error when attempting to use billing presets without passing a weights column to facilitate weighted least squares.


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
