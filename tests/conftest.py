""" py.test runs hooks etc. from conftest.py files. they apply recursively to test suites below.

py.test docs: https://docs.pytest.org/en/2.7.3/plugins.html
"""
from eemeter.log import setup_logging, disable_file_logging

def pytest_configure():
    setup_logging(eemeter_log_level='DEBUG', allow_console_logging=True)
    disable_file_logging()
