from eemeter import (
    EEMeterError,
    NoBaselineDataError,
    NoReportingDataError,
    MissingModelParameterError,
    UnrecognizedModelTypeError,
)

import pytest


def test_eemeter_error():
    with pytest.raises(EEMeterError):
        raise EEMeterError


def test_no_baseline_data_error():
    with pytest.raises(NoBaselineDataError):
        raise NoBaselineDataError
    assert isinstance(NoBaselineDataError(), EEMeterError)


def test_no_reporting_data_error():
    with pytest.raises(NoReportingDataError):
        raise NoReportingDataError
    assert isinstance(NoReportingDataError(), EEMeterError)


def test_missing_model_parameter_error():
    with pytest.raises(MissingModelParameterError):
        raise MissingModelParameterError
    assert isinstance(MissingModelParameterError(), EEMeterError)


def test_unrecognized_model_type_error():
    with pytest.raises(UnrecognizedModelTypeError):
        raise UnrecognizedModelTypeError
    assert isinstance(UnrecognizedModelTypeError(), EEMeterError)
