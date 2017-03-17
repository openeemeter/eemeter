import numpy as np
import pandas as pd

from eemeter.modeling.models.elastic_net_base import ElasticNetCVBaseModel


class BillingElasticNetCVModel(ElasticNetCVBaseModel):
    ''' Linear regression of energy values against CDD/HDD with elastic net
    regularization.

    Parameters
    ----------
    cooling_base_temp : float
        Base temperature (degrees F) used in calculating cooling degree days.
    heating_base_temp : float
        Base temperature (degrees F) used in calculating heating degree days.
    n_bootstrap : int
        Number of points to exclude during bootstrap error estimation.
    '''

    def __init__(self, cooling_base_temp=65, heating_base_temp=65,
                 n_bootstrap=100, modeling_period_interpretation='baseline'):

        super(BillingElasticNetCVModel, self).__init__(
            cooling_base_temp, heating_base_temp, n_bootstrap)
        self.modeling_period_interpretation = modeling_period_interpretation

    def __repr__(self):
        return (
            'BillingElasticNetCVModel(cooling_base_temp={},'
            ' heating_base_temp={}, n_bootstrap={})'
            .format(
                self.cooling_base_temp,
                self.heating_base_temp,
                self.n_bootstrap
            )
        )

    def _model_data_from_input_data(self, input_data):
        trace_data, temperature_data = input_data

        cdd = self._cdd(temperature_data)
        hdd = self._hdd(temperature_data)

        model_data = pd.DataFrame({
            'energy': trace_data.iloc[:-1],
            'CDD': cdd,
            'HDD': hdd
        }, columns=['energy', 'CDD', 'HDD'])

        model_data = model_data.dropna()
        return model_data

    def _cdd(self, temperature_data):
        if 'hourly' in temperature_data.index.names:
            cdd = np.maximum(temperature_data - self.cooling_base_temp, 0.0)\
                             .groupby(level='period').sum()[0] / 24.0
        elif 'daily' in temperature_data.index.names:
            cdd = np.maximum(temperature_data - self.cooling_base_temp, 0.0)\
                             .groupby(level='period').sum()[0]
        else:
            cdd = []
        return cdd

    def _hdd(self, temperature_data):
        if 'hourly' in temperature_data.index.names:
            hdd = np.maximum(self.heating_base_temp - temperature_data, 0.0)\
                             .groupby(level='period').sum()[0] / 24.0
        elif 'daily' in temperature_data.index.names:
            hdd = np.maximum(self.heating_base_temp - temperature_data, 0.0)\
                             .groupby(level='period').sum()[0]
        else:
            hdd = []
        return hdd

    def _patsy_formula(self, model_data):
        return self.base_formula

    def _model_data_from_demand_fixture_data(self, demand_fixture_data):

        # needs only tempF
        model_data = demand_fixture_data.resample(
            pd.tseries.frequencies.Day()).agg({'tempF': np.mean})

        model_data.loc[:, 'CDD'] = np.maximum(model_data.tempF -
                                              self.cooling_base_temp, 0.)
        model_data.loc[:, 'HDD'] = np.maximum(self.heating_base_temp -
                                              model_data.tempF, 0.)

        return model_data
