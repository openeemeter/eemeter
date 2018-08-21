API Docs
========

Results
-------

.. autoclass:: eemeter.CandidateModel
   :members:

.. autoclass:: eemeter.DataSufficiency
   :members:

.. autoclass:: eemeter.ModelFit
   :members:


CalTRACK methods
----------------

.. autofunction:: eemeter.caltrack_method

.. autofunction:: eemeter.caltrack_sufficiency_criteria

.. autofunction:: eemeter.caltrack_metered_savings

.. autofunction:: eemeter.caltrack_modeled_savings

.. autofunction:: eemeter.caltrack_predict

.. autofunction:: eemeter.get_too_few_non_zero_degree_day_warning

.. autofunction:: eemeter.get_total_degree_day_too_low_warning

.. autofunction:: eemeter.get_parameter_negative_warning

.. autofunction:: eemeter.get_parameter_p_value_too_high_warning

.. autofunction:: eemeter.get_single_cdd_only_candidate_model

.. autofunction:: eemeter.get_single_hdd_only_candidate_model

.. autofunction:: eemeter.get_single_cdd_hdd_candidate_model

.. autofunction:: eemeter.get_intercept_only_candidate_models

.. autofunction:: eemeter.get_cdd_only_candidate_models

.. autofunction:: eemeter.get_hdd_only_candidate_models

.. autofunction:: eemeter.get_cdd_hdd_candidate_models

.. autofunction:: eemeter.select_best_candidate


Data transformation utilities
-----------------------------

.. autofunction:: eemeter.as_freq

.. autofunction:: eemeter.compute_temperature_features

.. autofunction:: eemeter.day_counts

.. autofunction:: eemeter.get_baseline_data

.. autofunction:: eemeter.get_reporting_data

.. autofunction:: eemeter.merge_temperature_data

.. autofunction:: eemeter.remove_duplicates


Data loading
------------

.. autofunction:: eemeter.meter_data_from_csv

.. autofunction:: eemeter.meter_data_from_json

.. autofunction:: eemeter.meter_data_to_csv

.. autofunction:: eemeter.temperature_data_from_csv

.. autofunction:: eemeter.temperature_data_from_json

.. autofunction:: eemeter.temperature_data_to_csv


Sample Data
-----------

.. autofunction:: eemeter.samples

.. autofunction:: eemeter.load_sample


Visualization
-------------

.. autofunction:: eemeter.plot_time_series

.. autofunction:: eemeter.plot_energy_signature


Warnings
--------

.. autoclass:: eemeter.EEMeterWarning
   :members:


Exceptions
----------

.. autoexception:: eemeter.EEMeterError

.. autoexception:: eemeter.NoBaselineDataError

.. autoexception:: eemeter.NoReportingDataError

.. autoexception:: eemeter.MissingModelParameterError

.. autoexception:: eemeter.UnrecognizedModelTypeError


Version
-------
.. autofunction:: eemeter.get_version
