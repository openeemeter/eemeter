import pytest
import pandas as pd

from eemeter.modeling.models import GAS_ENERGY, ELECTRICITY_ENERGY
from eemeter.modeling.models.savings import SavingsPredictionModel

GAS_ENERGY = 'NATURAL_GAS_CONSUMPTION_SUPPLIED'
ELECTREICITY_ENERGY = 'ELECTRICITY_CONSUMPTION_SUPPLIED'

@pytest.fixture
def dummy_electricty_saving_data():
    """
    Dummy data on annualized savings for electricity
    Returns
    -------
     electricity_savings_kwh = intercept_coefficient + 0.5 * heating_coefficient +
                               0.5 * cooling_coefficient
    """
    electric_saving = pd.DataFrame({
        'electricity_savings_kwh' : [4.0, 1.0],
        'intercept_coefficient' : [1.0, 1.0],
        'heating_coefficient' : [2.0, -1.0],
        'cooling_coefficient' : [4.0, 1.0]
    })
    return electric_saving

@pytest.fixture
def dummy_gas_saving_data():
    """
         electricity_savings_kwh = intercept_coefficient + 0.5 * heating_coefficient
    """
    electric_saving = pd.DataFrame({
        'natural_gas_savings_thm' : [2.0, 0.5],
        'intercept_coefficient' : [1.0, 1.0],
        'heating_coefficient' : [2.0, -1.0],
    })
    return electric_saving

def test_fit(dummy_gas_saving_data, dummy_electricty_saving_data):
    model = SavingsPredictionModel(energy_type=GAS_ENERGY)
    model_wts = model.fit(dummy_gas_saving_data)
    assert model_wts is not None

    model = SavingsPredictionModel(energy_type=ELECTRICITY_ENERGY)
    model_wts = model.fit(dummy_electricty_saving_data)
    assert model_wts is not None

def test_predict_with_model_weights(dummy_gas_saving_data):
    model_weight = {
        'Intercept' : 0.5,
        'intercept_coefficient' : 0.05,
        'heating_coefficient' : 0.05
    }
    model = SavingsPredictionModel(energy_type=GAS_ENERGY,
                                   model_weights=model_weight)

    # Drop the column which needs to be predicted.
    # model.predict_with_model_weights expects exactly the
    # columns which are features to be used for prediction.
    data = dummy_gas_saving_data.drop('natural_gas_savings_thm', 1)
    for index, row in data.iterrows():
        prediction = model.predict_with_model_weights(row.to_dict())
        assert type(prediction) == float

def test_out_of_sample_stats(dummy_gas_saving_data):
    model = SavingsPredictionModel(energy_type=GAS_ENERGY)
    model.fit(dummy_gas_saving_data)
    stats = model.out_of_sample_stats(dummy_gas_saving_data)
    assert 'rmse' in stats
    assert 'savings_precision' in stats


