import pytest
import os

@pytest.fixture(params=["consumptions0.csv",
                        "consumptions1.csv"])
def consumption_csv_filename(request):
    return os.path.join(os.path.dirname(__file__),'resources',request.param)

@pytest.fixture(params=["consumptions0.xlsx"])
def consumption_xlsx_filename(request):
    return os.path.join(os.path.dirname(__file__),'resources',request.param)
