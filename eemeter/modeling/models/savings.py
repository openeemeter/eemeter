import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error
from math import sqrt

from eemeter.modeling.models import GAS_ENERGY, ELECTRICITY_ENERGY


class SavingsPredictionModel:
    """
    A simple linear regression  model to predict savings for potential
    clients.
    We build a simple linear regression model from our past savings data which
    has following information for each traces
    1. heating_coefficient, 2. cooling_coefficient, 3. intercept_coef
    4. annualized savings in Kwh
    The goal is to learn weights of coeficients in predicting savings.
    For new clients, we will build our usual Caltrack based models to learn
    coeficients. We can then use the coeficient and weights learned from this
    model to predict annualized savings for client.
    Parameters
    ----------
    energy_type : GAS/Electrcity, Regression formula is dependent on the energy type
    response_var_name: Dependent variable name
    model_weights: You can also directly init this class with model weights, e.g:
    { 'intercept_coefficient' : 0.05, 'heating_coefficient' : 0.09 }
    formula:  Regression formula
    """
    def __init__(self,
                 energy_type,
                 response_var_name=None,
                 model_weights=None,
                 formula=None):
        self.model_weights = model_weights
        self.energy_type = energy_type
        self.response_var_name = response_var_name
        if formula:
            self.formula = formula
        elif energy_type == GAS_ENERGY:
            self.formula = "natural_gas_savings_thm ~ heating_coefficient + intercept_coefficient"
            self.response_var_name = 'natural_gas_savings_thm'
        elif energy_type == ELECTRICITY_ENERGY:
            self.formula = "electricity_savings_kwh ~ cooling_coefficient +\
            heating_coefficient + intercept_coefficient "
            self.response_var_name = 'electricity_savings_kwh'
        else:
            raise ValueError("")

        self.fitted_model = None
        self.model_obj = None

    def predict(self, df):
        """
        Parameters
        ----------
        df :Dataframe with columns expected as in self.formula
        Returns
        -------
        """
        pred = self.fitted_model.predict(df)
        # Series
        return pred

    def predict_with_model_weights(self, feature_values):
        """
        Predicts when this class is init with model weights.
        Use feature values and models weights to make prediction.
        Parameters
        ----------
        feature_values (dict): The keys in feature_values
        should be exactly same as the keys in self.model_weights
        except the Intercept, which is present in model_weights.
        Returns: Prediction (float)
        -------
        """
        if not self.model_weights:
            raise ValueError("Model Weights Not Set")

        if 'Intercept' not in self.model_weights:
            raise ValueError("Intercept not present in model weights")

        model_weight_keys = [xx for xx in
                             self.model_weights.keys() if xx != "Intercept"]

        if set(model_weight_keys) != set(feature_values.keys()):
            raise ValueError("Feature Keys and Model Weights Keys do not match")

        response = self.model_weights['Intercept']
        for feature_key, value in feature_values.items():
            feature_weight = self.model_weights[feature_key]
            response += (value * feature_weight)

        return response

    def out_of_sample_stats(self, df, response_var_name=None):
        """
        Return model stats: rmse, savings_precision
        Parameters
        ----------
        df
        response_var_name

        Returns
        -------

        """
        prediction = self.predict(df)
        if response_var_name is None:
            reponse_var_name = self.response_var_name
        actual = df[reponse_var_name]
        rmse = sqrt(mean_squared_error(prediction, actual))

        savings = 0.0
        for idx, predicted_saving in enumerate(prediction):
            if predicted_saving > 0.0 and actual.iloc[idx] > 0.0:
                savings += 1.0

        savings_precision = savings / len(df)
        return {'rmse': rmse,
                'savings_precision': savings_precision}

    def fit(self, data_frame):
        """
        Fits model. Required: self.formula to defined and
        data_frame should have same column as expected by
        self.formula
        Parameters
        ----------
        data_frame : Pandas Data Frame with column as expected
        by self.formula.
        Returns : Dict of model weights
        -------
        """
        ols_model = smf.ols(formula=self.formula, data=data_frame)
        self.fitted_model = ols_model.fit()
        self.model_obj = ols_model
        self.model_weights = self.fitted_model.params.to_dict()
        return self.model_weights
