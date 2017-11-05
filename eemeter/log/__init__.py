# -*- coding: utf-8 -*-
""" boilerplate to set up logging cleanly and avoid redundant setup (avoid unintended side-effects)

note, this module is intentionally not called `logging`, as that can mask stdlib `logging` module.
at least in certain contexts like PyCharm's test runner. leave it as `log` for best results.
"""
import json
import logging
import logging.config
import os

HERE = os.path.abspath(os.path.dirname(__file__))
DEFAULT_LOGGING_CONFIG_PATH = os.path.join(HERE, "logging.json")

EEMETER_LOGGER_NAME = 'eemeter'  # This needs to match the top package name.
EEMETER_LOGGER_HAS_BEEN_SET_UP = False


def setup_logging(
        path_to_logging_json=DEFAULT_LOGGING_CONFIG_PATH,
        eemeter_log_level=None,
        allow_console_logging=True):
    """ Setup logging configuration from file.

    Parameters
    ----------
    path_to_logging_json : str
        Optional path to custom logging configuration json file. passed to dictConfig()
    eemeter_log_level : str
        Optional override for eemeter logging level, such as "DEBUG". (Cascades to eemeter.*)
    allow_console_logging : bool
        If not True, then all logging handlers with 'console' in the name will be disabled.
    """
    global EEMETER_LOGGER_HAS_BEEN_SET_UP  # global is evil in general, but this is OK.
    our_logger = logging.getLogger(EEMETER_LOGGER_NAME)

    with open(path_to_logging_json, 'r') as f:
        config = json.load(f)

    logging.config.dictConfig(config)

    our_logger.debug('setting logging.captureWarnings(True)')
    logging.captureWarnings(True)

    if eemeter_log_level is not None:
        try:
            our_logger.setLevel(logging.getLevelName(eemeter_log_level.upper()))
        except ValueError:
            our_logger.error(
                    "could not set logging level to {!r}. "
                    "please use a standard Python logging level.".format(eemeter_log_level))
            raise

    if not allow_console_logging:
        for _logger in (our_logger, logging.getLogger('py.warnings')):
            handlers = [h for h in _logger.handlers if 'console' not in h.get_name().lower()]
            _logger.handlers = handlers

    EEMETER_LOGGER_HAS_BEEN_SET_UP = True
    return our_logger


def disable_file_logging():
    """ Disable all logging handlers that appear to be FileHandler variants.

    A utility mainly for test suite, where we prefer console output, as py.test captures all
    console output then presents it when relevant; and to avoid mixing normal & test logs.
    """
    loggers = logging.Logger.manager.loggerDict.values()
    for logger in loggers:
        if hasattr(logger, 'log'):  # skip logging.PlaceHolder instances, etc.
            handlers = []
            for handler in logger.handlers:
                if not handler:
                    pass
                elif issubclass(handler.__class__, logging.FileHandler):
                    pass
                elif 'file' in (handler.get_name() or '').lower():
                    pass
                else:
                    handlers.append(handler)
            logger.handlers = handlers
