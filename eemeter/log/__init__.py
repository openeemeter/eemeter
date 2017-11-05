# -*- coding: utf-8 -*-
""" boilerplate to set up logging cleanly and avoid redundant setup (avoid unintended side-effects)

note, this module is intentionally not called `logging`, as that can mask stdlib `logging` module.
at least in certain contexts like PyCharm's test runner. leave it as `log` for best results.
"""
import json
import logging
import logging.config
import os
import sys
import warnings


HERE = os.path.abspath(os.path.dirname(__file__))

EEMETER_LOGGER_NAME = 'eemeter'  # This needs to match the top package name.
EEMETER_LOGGER_HAS_BEEN_SET_UP = False


def setup_logging(
        path_to_logging_json=os.path.join(HERE, "logging.json"),
        eemeter_log_level=None,
        allow_console_logging=True):
    """ Setup logging configuration from file.

    Parameters
    ----------
    path_to_logging_json : str
        Optional path to custom logging configuration json file. passed to dictConfig()
    eemeter_log_level : str
        Optional override for eemeter logging level. (Cascades to eemeter.*)
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
        except:
            our_logger.error("could not set logging level to {!r}. "
                         "please use a standard Python logging level.".format(eemeter_log_level))
            raise

    if not allow_console_logging:
        for _logger in (our_logger, logging.getLogger('py.warnings')):
            handlers = [h for h in _logger.handlers if not 'console' in h.get_name().lower()]
            _logger.handlers = handlers

    EEMETER_LOGGER_HAS_BEEN_SET_UP = True
    return our_logger
