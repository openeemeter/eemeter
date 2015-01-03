import pytest

def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", help="run slow tests")
    parser.addoption("--no-internet", action="store_true", help="skip tests which require internet")

def pytest_runtest_setup(item):
    if 'slow' in item.keywords and not item.config.getoption("--runslow"):
        pytest.skip("need --runslow option to run")
    if 'internet' in item.keywords and item.config.getoption("--no-internet"):
        pytest.skip("skipping because of --no-internet flag")
