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
    ModelFit,
)
from .caltrack import (
    caltrack_daily_method,
    caltrack_daily_sufficiency_criteria,
    predict_caltrack_daily,
    get_too_few_non_zero_degree_day_warning,
    get_total_degree_day_too_low_warning,
    get_parameter_negative_warning,
    get_parameter_p_value_too_high_warning,
    get_intercept_only_candidate_models,
    get_cdd_only_candidate_models,
    get_hdd_only_candidate_models,
    get_cdd_hdd_candidate_models,
    select_best_candidate,
)
from .exceptions import (
    EEMeterError,
    NoBaselineDataError,
    NoReportingDataError,
    MissingModelParameterError,
    UnrecognizedModelTypeError,
)
from .transform import (
    billing_as_daily,
    get_baseline_data,
    get_reporting_data,
    merge_temperature_data,
    day_counts,
)
from .io import (
    meter_data_from_csv,
    meter_data_from_json,
    meter_data_to_csv,
    temperature_data_from_csv,
    temperature_data_from_json,
    temperature_data_to_csv,
)
from .visualization import (
    plot_candidate,
    plot_energy_signature,
    plot_model_fit,
    plot_time_series,
)
from .samples.load import (
    samples,
    load_sample,
)

def get_version():
    return __version__


# Set default logging handler to avoid "No handler found" warnings.
from logging import NullHandler

logging.getLogger(__name__).addHandler(NullHandler())
