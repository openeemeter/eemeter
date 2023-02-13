.. _api-docs:

API Docs
========

.. _caltrack:

CalTRACK
--------

CalTRACK design matrix creation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These functions are designed as shortcuts to common CalTRACK design matrix inputs.

.. autofunction:: eemeter.create_caltrack_hourly_preliminary_design_matrix

.. autofunction:: eemeter.create_caltrack_hourly_segmented_design_matrices

.. autofunction:: eemeter.create_caltrack_daily_design_matrix

.. autofunction:: eemeter.create_caltrack_billing_design_matrix

.. _caltrack-hourly-api:

CalTRACK Hourly
~~~~~~~~~~~~~~~

These classes and functions are designed to assist with running the CalTRACK Hourly
methods. See also :ref:`caltrack-hourly-quickstart`.

.. autoclass:: eemeter.CalTRACKHourlyModel
   :members:

.. autoclass:: eemeter.CalTRACKHourlyModelResults
   :members:

.. autofunction:: eemeter.caltrack_hourly_fit_feature_processor

.. autofunction:: eemeter.caltrack_hourly_prediction_feature_processor

.. autofunction:: eemeter.fit_caltrack_hourly_model_segment

.. autofunction:: eemeter.fit_caltrack_hourly_model

.. autofunction:: eemeter.eemeter_hourly

.. _caltrack-billing-daily-api:

CalTRACK Daily and Billing (Usage per Day)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These classes and functions are designed to assist with running the CalTRACK Daily
and Billing methods. See also :ref:`caltrack-billing-daily-quickstart`.

.. autoclass:: eemeter.CalTRACKUsagePerDayCandidateModel
   :members:

.. autoclass:: eemeter.CalTRACKUsagePerDayModelResults
   :members:

.. autoclass:: eemeter.DataSufficiency
   :members:

.. autoclass:: eemeter.ModelPrediction
   :members:

.. autofunction:: eemeter.fit_caltrack_usage_per_day_model

.. autofunction:: eemeter.caltrack_sufficiency_criteria

.. autofunction:: eemeter.caltrack_usage_per_day_predict

.. autofunction:: eemeter.plot_caltrack_candidate

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

.. autofunction:: eemeter.eemeter_daily

Savings
-------

These methods are designed for computing metered and normal year savings.

.. autofunction:: eemeter.metered_savings

.. autofunction:: eemeter.modeled_savings


Exceptions
----------

These exceptions are used in the package to indicate various common issues.

.. autoexception:: eemeter.EEMeterError

.. autoexception:: eemeter.NoBaselineDataError

.. autoexception:: eemeter.NoReportingDataError

.. autoexception:: eemeter.MissingModelParameterError

.. autoexception:: eemeter.UnrecognizedModelTypeError


Features
--------

These methods are used to compute features that are used in creating CalTRACK models.

.. autofunction:: eemeter.compute_usage_per_day_feature

.. autofunction:: eemeter.compute_occupancy_feature

.. autofunction:: eemeter.compute_temperature_features

.. autofunction:: eemeter.compute_temperature_bin_features

.. autofunction:: eemeter.compute_time_features

.. autofunction:: eemeter.estimate_hour_of_week_occupancy

.. autofunction:: eemeter.fit_temperature_bins

.. autofunction:: eemeter.get_missing_hours_of_week_warning

.. autofunction:: eemeter.merge_features


Input and Output Utilities
--------------------------

These functions are used for reading and writing meter and temperature data.

.. autofunction:: eemeter.meter_data_from_csv

.. autofunction:: eemeter.meter_data_from_json

.. autofunction:: eemeter.meter_data_to_csv

.. autofunction:: eemeter.temperature_data_from_csv

.. autofunction:: eemeter.temperature_data_from_json

.. autofunction:: eemeter.temperature_data_to_csv


Metrics
-------

This class is used for computing model metrics.

.. autoclass:: eemeter.ModelMetrics
   :members:


Sample Data
-----------

These sample data are provided to make things easier for new users.

.. autofunction:: eemeter.samples

.. autofunction:: eemeter.load_sample


Segmentation
------------

These methods are used within CalTRACK hourly to support building multiple partial
models and combining them into one full model.

.. autofunction:: eemeter.iterate_segmented_dataset

.. autofunction:: eemeter.segment_time_series

.. autoclass:: eemeter.CalTRACKSegmentModel
   :members:

.. autoclass:: eemeter.SegmentedModel
   :members:

.. autoclass:: eemeter.HourlyModelPrediction
   :members:


Transformation utilities
------------------------

These functions are used to various common data transformations based on pandas inputs.

.. autofunction:: eemeter.as_freq

.. autofunction:: eemeter.day_counts

.. autofunction:: eemeter.get_baseline_data

.. autofunction:: eemeter.get_reporting_data

.. autoclass:: eemeter.Term
   :members:

.. autofunction:: eemeter.get_terms

.. autofunction:: eemeter.remove_duplicates

.. autofunction:: eemeter.overwrite_partial_rows_with_nan

.. autofunction:: eemeter.format_temperature_data_for_eemeter

.. autofunction:: eemeter.format_energy_data_for_eemeter

.. autofunction:: eemeter.sum_gas_and_elec

.. autofunction:: eemeter.trim

.. autofunction:: eemeter.add_freq

Version
-------

This method can used to verify the eemeter version.

.. autofunction:: eemeter.get_version


Visualization
-------------

These functions are used to visualization of models and meter and temperature data
inputs.

.. autofunction:: eemeter.plot_time_series

.. autofunction:: eemeter.plot_energy_signature


Warnings
--------

.. autoclass:: eemeter.EEMeterWarning
   :members:
