import pandas as pd


def _compute_r_squared(combined):
    
    r_squared = (combined[['predicted','observed']].corr().iloc[0,1]**2)
    
    return r_squared

def _compute_r_squared_adj(r_squared, length, regressors):

    r_squared_adj = (1 - (1 - r_squared)*(length - 1)/ \
            (length - regressors - 1)) 

    return r_squared_adj

def _compute_cvrmse(combined):
    
    cvrmse = ((combined['residuals']**2).mean()**0.5)/ \
        (combined['observed'].mean())
    
    return cvrmse

def _compute_mape(combined):
    
    mape = (combined['residuals'] / combined['observed']).abs().mean()
    
    return mape

def _compute_nmae(combined):
    
    nmae = (combined['residuals'].abs().sum())/(combined['observed'].sum())

    return nmae

def _compute_nmbe(combined):
    
    nmbe = (combined['residuals'].sum()/combined['observed'].sum())
    
    return nmbe

def _compute_autocorr_resid(combined, autocorr_lags):
    
    autocorr_resid = (combined['residuals'].autocorr(lag=autocorr_lags))
    
    return autocorr_resid

def _compute_model_uncertainty():
    
    return model_uncertainty

class ModelMetrics(object):
    ''' Contains information on a variety of measures of model fit. Different 
    from ModelFit object in that only one of ModelFit's outputs is a
    measure of fit (r-squared), and ModelFit contains results relating to 
    failed models (and not just the one ultimately selected model). 
    ModelMetrics contains many measures of fit, and does so only for a single
    model. 

   Parameters
    ----------
    observed_input : :any:`pandas.Series`
        Series with :any:`pandas.DatetimeIndex` with a set of electricity or 
        gas meter values.
    predicted_input : :any:`pandas.Series`
        Series with :any:`pandas.DatetimeIndex` with a set of electricity or 
        gas meter values.
    frequency : :any:`str`
        The frequency of the two input data series. Options are 
        `'hourly'`, `'daily'`, `'billing_monthly'`, `'billing_bimonthly'`, or
        `'other'`. Determines whether model uncertainty is calculated 
        (presently only available for daily- and billing-period data).
    '''

    def __init__(
        self, observed_input, predicted_input, frequency='other', regressors=1, autocorr_lags=1
    ):
        observed_input.name = "observed"
        predicted_input.name = "predicted"
        
        observed = observed_input.to_frame()
        predicted = predicted_input.to_frame()
        
        self.observed_length = observed.shape[0]
        self.predicted_length = predicted.shape[0]
        
        warnings = []
        # warn if the input series have different lengths
        if self.observed_length != self.predicted_length:
            warnings.append(EEMeterWarning(
                qualified_name='eemeter.ModelMetrics.observed_and_predicted_lengths_differ',
                description=(
                    'Input data series -- observed and predicted -- are different lengths.'
                ),
                data={
                    0
                        # ASK PHIL -- maybe just self.observed_length self.predicted_length
                },
            ))

        combined = observed.merge(predicted, left_index=True, \
                        right_index=True)
        
        # Calculate residuals because these are an input for most of the metrics  
        combined['residuals'] = (combined.predicted - combined.observed)
        
        self.freq = frequency
        self.numregressors = regressors
        self.autocorr_lags = autocorr_lags
        
        # Calculate and record mean
        self.observed_mean = combined['observed'].mean()
        self.predicted_mean = combined['predicted'].mean()
        
        # Calculate and record skew
        self.observed_skew = combined['observed'].skew()
        self.predicted_skew = combined['predicted'].skew()
        
        # Calculate and record excess kurtosis
        self.observed_kurtosis = combined['observed'].kurtosis()
        self.predicted_kurtosis = combined['predicted'].kurtosis()
        
        # Calculate and record coefficient of variation standard deviation 
        # (CVSTD)
        self.observed_cvstd = combined['observed'].std()/ \
            self.observed_mean
        
        self.predicted_cvstd = combined['predicted'].std()/ \
            self.predicted_mean
        
        # Calculate and record r-squared
        self.r_squared = _compute_rsquared(combined)
        
        # Calculate and record number of observations
        self.merged_length = combined.shape[0]
        
        # Calculate and record adjusted r-squared. NOTE: I can't get this to 
        # exactly match the ModelFit adjusted r-squared, though I could match 
        # it to five decimal places if I reduced the number of regressors by 
        # two in the denominator instead of one
        self.r_squared_adj = _compute_r_squared_adj(self.r_squared, \
            self.merged_length, self.numregressors)
        
        self.cvrmse = _compute_cvrmse(combined)
        
        self.mape = _compute_mape(combined)
        
        # Because MAPE will often be infinite (since meter values, the
        # denominator, can hit zero), create a new DataFrame with all rows
        # removed where meter_value is zero
        nometerzeros = combined[combined['observed'] > 0]
        
        # Calculate and record MAPE with all rows excluded where meter_value is
        # zero
        self.mape_nozeros = _compute_mape(nometerzeros)
        
        # Calculate and record the number of times meter_value is zero. We 
        # subtract one from length because length ends in a nan value that
        # nometerzeros didn't include
        self.nummeterzeros = ((self.merged_length - 1) - nometerzeros.shape[0])
        
        # Calculate and record normalized mean absolute error (NMAE)
        self.nmae = _compute_nmae(combined)
        
        # Calculate and record normalized mean bias error (NMBE)
        self.nmbe = _compute_nmbe(combined)
                
        # Calculate and record autocorrelation coefficient of residuals
        self.autocorr_resid = _compute_autocorr_resid(combined, autocorr_lags)
        
        if frequency == 'daily':
            self.model_uncertainty = _compute_model_uncertainty()
            
        elif frequency == 'billing_monthly':
            self.model_uncertainty = _compute_model_uncertainty()


    # CREATE JSON DUMP code

        































