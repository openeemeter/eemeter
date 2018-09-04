import pandas as pd


__all__ = (
    'ModelMetrics',
)


def _compute_r_squared(combined):

    r_squared = (combined[['predicted','observed']].corr().iloc[0, 1] ** 2)

    return r_squared


def _compute_r_squared_adj(r_squared, length, num_parameters):

    r_squared_adj = (1 - (1 - r_squared) * (length - 1) /
        (length - num_parameters - 1))

    return r_squared_adj


def _compute_rmse(combined):

    rmse = ((combined['residuals'].astype(float) ** 2).mean() ** 0.5)

    return rmse


def _compute_rmse_adj(combined, length, num_parameters):

    rmse_adj = (((combined['residuals'].astype(float) ** 2).sum() /
        (length - num_parameters)) ** 0.5)

    return rmse_adj


def _compute_cvrmse(rmse, observed_mean):

    cvrmse = rmse / observed_mean

    return cvrmse


def _compute_cvrmse_adj(rmse_adj, observed_mean):

    cvrmse_adj = rmse_adj / observed_mean

    return cvrmse_adj


def _compute_mape(combined):

    mape = (combined['residuals'] / combined['observed']).abs().mean()

    return mape


def _compute_nmae(combined):

    nmae = ((combined['residuals'].astype(float).abs().sum()) /
        (combined['observed'].sum()))

    return nmae


def _compute_nmbe(combined):

    nmbe = (combined['residuals'].astype(float).sum() /
        combined['observed'].sum())

    return nmbe


def _compute_autocorr_resid(combined, autocorr_lags):

    autocorr_resid = (combined['residuals'].autocorr(lag=autocorr_lags))

    return autocorr_resid


class ModelMetrics(object):
    ''' Contains measures of model fit and summary statistics on the input series.

    Parameters
    ----------
    observed_input : :any:`pandas.Series`
        Series with :any:`pandas.DatetimeIndex` with a set of electricity or
        gas meter values.
    predicted_input : :any:`pandas.Series`
        Series with :any:`pandas.DatetimeIndex` with a set of electricity or
        gas meter values.
    num_parameters : :any:`int`, optional
        The number of parameters (excluding the intercept) used in the
        regression from which the predictions were derived.
    autocorr_lags : :any:`int`, optional
        The number of lags to use when calculating the autocorrelation of the
        residuals

    Attributes
    ----------
    observed_length : :any:`int`
        The length of the observed_input series.
    predicted_length : :any:`int`
        The length of the predicted_input series.
    merged_length : :any:`int`
        The length of the dataframe resulting from the inner join of the
        observed_input series and the predicted_input series.
    observed_mean : :any:`float`
        The mean of the observed_input series.
    predicted_mean : :any:`float`
        The mean of the predicted_input series.
    observed_skew : :any:`float`
        The skew of the observed_input series.
    predicted_skew : :any:`float`
        The skew of the predicted_input series.
    observed_kurtosis : :any:`float`
        The excess kurtosis of the observed_input series.
    predicted_kurtosis : :any:`float`
        The excess kurtosis of the predicted_input series.
    observed_cvstd : :any:`float`
        The coefficient of standard deviation of the observed_input series.
    predicted_cvstd : :any:`float`
        The coefficient of standard deviation of the predicted_input series.
    r_squared : :any:`float`
        The r-squared of the model from which the predicted_input series was
        produced.
    r_squared_adj : :any:`float`
        The r-squared of the predicted_input series relative to the
        observed_input series, adjusted by the number of parameters in the model.
    cvrmse : :any:`float`
        The coefficient of variation (root-mean-squared error) of the
        predicted_input series relative to the observed_input series.
    cvrmse_adj : :any:`float`
        The coefficient of variation (root-mean-squared error) of the
        predicted_input series relative to the observed_input series, adjusted
        by the number of parameters in the model.
    mape : :any:`float`
        The mean absolute percent error of the predicted_input series relative
        to the observed_input series.
    mape_no_zeros : :any:`float`
        The mean absolute percent error of the predicted_input series relative
        to the observed_input series, with all time periods dropped where the
        observed_input series was not greater than zero.
    num_meter_zeros : :any:`int`
        The number of time periods for which the observed_input series was not
        greater than zero.
    nmae : :any:`float`
        The normalized mean absolute error of the predicted_input series
        relative to the observed_input series.
    nmbe : :any:`float`
        The normalized mean bias error of the predicted_input series relative
        to the observed_input series.
    autocorr_resid : :any:`float`
        The autocorrelation of the residuals (where the residuals equal the
        predicted_input series minus the observed_input series), measured
        using a number of lags equal to autocorr_lags.
    '''

    def __init__(
        self, observed_input, predicted_input, num_parameters = 1, autocorr_lags = 1
    ):
        if (num_parameters < 0):
            raise ValueError('num_parameters must be greater than or equal to zero')
        if (autocorr_lags <= 0):
            raise ValueError('autocorr_lags must be greater than zero')

        observed = observed_input.to_frame().dropna()
        predicted = predicted_input.to_frame().dropna()
        observed.columns = ['observed']
        predicted.columns = ['predicted']

        self.observed_length = observed.shape[0]
        self.predicted_length = predicted.shape[0]

        if (self.observed_length != self.predicted_length) :
            raise ValueError('Input series are of different lengths')

        # Do an inner join on the two input series to make sure that we only
        # use observations with the same time stamps.
        combined = observed.merge(predicted, left_index=True,
            right_index=True)

        self.merged_length = combined.shape[0]

        # Calculate residuals because these are an input for most of the metrics.
        combined['residuals'] = (combined.predicted - combined.observed)

        self.num_parameters = num_parameters
        self.autocorr_lags = autocorr_lags

        self.observed_mean = combined['observed'].mean()
        self.predicted_mean = combined['predicted'].mean()

        self.observed_variance = combined['observed'].var(ddof=0)
        self.predicted_variance = combined['predicted'].var(ddof=0)

        self.observed_skew = combined['observed'].skew()
        self.predicted_skew = combined['predicted'].skew()

        self.observed_kurtosis = combined['observed'].kurtosis()
        self.predicted_kurtosis = combined['predicted'].kurtosis()

        self.observed_cvstd = combined['observed'].std() / self.observed_mean
        self.predicted_cvstd = combined['predicted'].std() / self.predicted_mean

        self.r_squared = _compute_r_squared(combined)
        self.r_squared_adj = _compute_r_squared_adj(self.r_squared,
            self.merged_length, self.num_parameters)

        self.rmse = _compute_rmse(combined)
        self.rmse_adj = _compute_rmse_adj(combined, self.merged_length,
            self.num_parameters)

        self.cvrmse = _compute_cvrmse(self.rmse, self.observed_mean)
        self.cvrmse_adj = _compute_cvrmse_adj(self.rmse_adj, self.observed_mean)

        # Create a new DataFrame with all rows removed where observed is
        # zero, so we can calculate a version of MAPE with the zeros excluded.
        # (Without the zeros excluded, MAPE becomes infinite when one observed
        # value is zero.)
        no_observed_zeros = combined[combined['observed'] > 0]

        self.mape = _compute_mape(combined)
        self.mape_no_zeros = _compute_mape(no_observed_zeros)

        self.num_meter_zeros = ((self.merged_length) - no_observed_zeros.shape[0])

        self.nmae = _compute_nmae(combined)

        self.nmbe = _compute_nmbe(combined)

        self.autocorr_resid = _compute_autocorr_resid(combined, autocorr_lags)

    def __repr__(self):
        return (
            "ModelMetrics(merged_length={}, r_squared_adj={}, cvrmse_adj={}, "
            "mape_no_zeros={}, nmae={}, nmbe={}, autocorr_resid={})"
            .format(
                self.merged_length, round(self.r_squared_adj, 3), round(self.cvrmse_adj, 3),
                round(self.mape_no_zeros, 3), round(self.nmae, 3), round(self.nmbe, 3),
                round(self.autocorr_resid, 3)
            )
        )

    def json(self):
        ''' Return a JSON-serializable representation of this result.

        The output of this function can be converted to a serialized string
        with :any:`json.dumps`.
        '''
        return {
            'observed_length': self.observed_length,
            'predicted_length': self.predicted_length,
            'merged_length': self.merged_length,
            'num_parameters': self.num_parameters,
            'observed_mean': self.observed_mean,
            'predicted_mean': self.predicted_mean,
            'observed_variance': self.observed_variance,
            'predicted_variance': self.predicted_variance,
            'observed_skew': self.observed_skew,
            'predicted_skew': self.predicted_skew,
            'observed_kurtosis': self.observed_kurtosis,
            'predicted_kurtosis': self.predicted_kurtosis,
            'observed_cvstd': self.observed_cvstd,
            'predicted_cvstd': self.predicted_cvstd,
            'r_squared': self.r_squared,
            'r_squared_adj': self.r_squared_adj,
            'rmse': self.rmse,
            'rmse_adj': self.rmse_adj,
            'cvrmse': self.cvrmse,
            'cvrmse_adj': self.cvrmse_adj,
            'mape': self.mape,
            'mape_no_zeros': self.mape_no_zeros,
            'num_meter_zeros': self.num_meter_zeros,
            'nmae': self.nmae,
            'nmbe': self.nmbe,
            'autocorr_resid': self.autocorr_resid,
        }
