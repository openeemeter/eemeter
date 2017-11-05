""" py.test runs hooks etc. from conftest.py files. they apply recursively to test suites below.

https://docs.pytest.org/en/3.2.3/writing_plugins.html?highlight=conftest
"""
from eemeter.log import setup_logging, disable_file_logging


def pytest_configure():
    setup_logging(eemeter_log_level='DEBUG', allow_console_logging=True)
    disable_file_logging()
