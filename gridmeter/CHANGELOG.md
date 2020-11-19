Changelog
=========

Development
-----------
* Rename to 'gridmeter' 


0.10.1
------


* Update description

0.10.0
------


* Rename to 'gridmeter' -- final release as eesampling

0.9.1
-----


* Add comparison pool equivalence to results_as_json().

0.9.0
-----

* Refactor equivalence calculation for bin selection (much faster)
* Change input format for equivalence in bin selection

0.8.0
-----

* Add synthetic data generation for testing and tutorials
* Add tutorial Jupyter notebook
* Rename Diagnostics --> StratifiedSamplingDiagnostics
* Expose all classes for top-level imports.
* Made adjustment to how n_samples_approx is calculated. It now works where the minimum sampled:treatment ratio can be violated if n_samples_approx is used as an upper bound and that upper bound is reached.

0.7.0
-----

* Rename train --> treatment, test --> pool 

0.6.1
-----

* Fix Github URL

0.6.0
-----

* First public release 
* Update default params for bin_selection.StratifiedSamplingBinSelector(...) so n_samples_approx = 5000 and relax_n_samples_approx_constraint=False and min_n_sampled_to_n_train_ratio = 0.25, which means that we aim for 5000 comparison group meters but if we can't reach it, we need at least 0.25 sample to train ratio or else it fails. 
* Add relax_n_samples_approx constraint so that you can use n_samples_approx as upper bound rather than a target.
* Refactor results_as_json a bit so selected sample output is cleaner.

0.5.5
-----

* Update results serialization.

0.5.4
-----

* Add kwargs and results serialization.

0.5.3
-----

* Separate bin selection into a different class.

0.5.2
-----

* Fix issue with naming during equivalence chisquare checking of diagnostics (this needs to be refactored later).

0.5.1
-----

* Renamed `min_bin_size` to `min_n_train_per_bin`.
* Move BinnedData to bins.py.
* Added chisquared equivalence option.
* Add equivalence via a separate dataframe.

0.5.0
-----

* Added some unit tests for modelling and some test framework.
* Generalized Diagnostics so that .plot_equivalence(...) can also plot the comparison pool.
* Changed automatic n_samples_approx to use the maximum number of samples available (based on how many test values are in the "worst" bin) rather than use binary search.
* Renamed n_outputs to n_samples_approx

0.4.2
-----

* Fix random seed so that numpy random seeding for pertubation is happening in the right place.
* Make a copy of the dataframe in the `_perturb()` function.

0.4.1
-----

* Add random seed option.

0.4.0
-----

* Support fixed-width or variable-with bins
* Auto-choose number of outputs via binary search

0.3.3
-----

* Scatter plot has fixed y scales and correct size


0.3.2
-----

* Fix bug if not using auto-bin

0.3.1
-----

* Remove plotly dependency

0.3.0
-----

* Simplify plotting
* Add auto_bin option


0.2.0
-----

* Big refactor, add plotting diagnostics.
* Add plotly support

0.1.0
-----

* Initial create of model.

0.0.1
-----

* Initial creation of library.
