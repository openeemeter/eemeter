__all__ = (
    "EEMeterError",
    "NoBaselineDataError",
    "NoReportingDataError",
    "MissingModelParameterError",
    "UnrecognizedModelTypeError",
)


class EEMeterError(Exception):
    """ Base class for EEmeter library errors.
    """

    pass


class NoBaselineDataError(EEMeterError):
    """ Error indicating lack of baseline data.
    """

    pass


class NoReportingDataError(EEMeterError):
    """ Error indicating lack of reporting data.
    """

    pass


class MissingModelParameterError(EEMeterError):
    """ Error indicating missing model parameter.
    """

    pass


class UnrecognizedModelTypeError(EEMeterError):
    """ Error indicating unrecognized model type.
    """

    pass
