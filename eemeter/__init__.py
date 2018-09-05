# -*- coding: utf-8 -*-
"""
eemeter library usage
~~~~~~~~~~~~~~~~~~~~~
The eemeter libary implements core Open Energy Efficiency metering methods.
Basic usage:
   >>> import eemeter
Full documentation is at <https://openee.io>.
:copyright: (c) 2018 by Open Energy Efficiency.
:license: Apache 2.0, see LICENSE for more details.
"""

import logging

from .__version__ import __title__, __description__, __url__, __version__
from .__version__ import __author__, __author_email__, __license__
from .__version__ import __copyright__
from .api import (
    CandidateModel,
    DataSufficiency,
    EEMeterWarning,
    ModelResults,
    HourlyModel,
)
from .caltrack import (
    caltrack_method,
    caltrack_sufficiency_criteria,
    caltrack_predict,
    get_single_cdd_only_candidate_model,
    get_single_hdd_only_candidate_model,
    get_single_cdd_hdd_candidate_model,
    get_cdd_hdd_candidate_models,
    get_cdd_only_candidate_models,
    get_hdd_only_candidate_models,
    get_intercept_only_candidate_models,
    get_parameter_negative_warning,
    get_parameter_p_value_too_high_warning,
    get_too_few_non_zero_degree_day_warning,
    get_total_degree_day_too_low_warning,
    plot_caltrack_candidate,
    select_best_candidate,
)
from .derivatives import metered_savings, modeled_savings
from .caltrack_hourly import (
    get_feature_hour_of_week,
    get_feature_occupancy,
    get_feature_binned_temperatures,
    get_design_matrix,
    caltrack_hourly_method,
    caltrack_hourly_predict,
    caltrack_hourly_fit_feature_processor,
    caltrack_hourly_prediction_feature_processor,
    fit_hourly_model_segment,
)
from .features import (
    compute_usage_per_day_feature,
    compute_occupancy_feature,
    compute_temperature_features,
    compute_temperature_bin_features,
    compute_time_features,
    estimate_hour_of_week_occupancy,
    fit_temperature_bins,
    merge_features,
)
from .segmentation import (
    iterate_segmented_dataset,
    segment_time_series,
    fit_segmented_model,
    SegmentModel,
    SegmentedModel,
)
from .exceptions import (
    EEMeterError,
    NoBaselineDataError,
    NoReportingDataError,
    MissingModelParameterError,
    UnrecognizedModelTypeError,
)
from .metrics import ModelMetrics
from .transform import (
    as_freq,
    day_counts,
    get_baseline_data,
    get_reporting_data,
    remove_duplicates,
    overwrite_partial_rows_with_nan,
)
from .io import (
    meter_data_from_csv,
    meter_data_from_json,
    meter_data_to_csv,
    temperature_data_from_csv,
    temperature_data_from_json,
    temperature_data_to_csv,
)
from .visualization import plot_energy_signature, plot_time_series
from .samples.load import samples, load_sample


def get_version():
    return __version__


# Set default logging handler to avoid "No handler found" warnings.
from logging import NullHandler

logging.getLogger(__name__).addHandler(NullHandler())
