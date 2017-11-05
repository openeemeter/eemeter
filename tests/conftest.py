""" py.test runs hooks etc. from conftest.py files. they apply recursively to test suites below.

py.test docs: https://docs.pytest.org/en/2.7.3/plugins.html
"""
import logging
from eemeter.log import setup_logging

def pytest_runtest_setup(item):
    setup_logging(eemeter_log_level=logging.DEBUG, allow_console_logging=True)