import numpy as np
import pandas as pd

# TODO : add test cases 
def pct_correction_factor(df : pd.DataFrame, observed : str = 'obs', counterfactual : str = 'predicted'):
    """
    Compute the difference in difference correction factor given a data frame and two columns.

    Parameters
    ----------
    df : Dataframe of comparison group 

    Returns
    -------
    Dataframe with correction factor added

    """

    if {observed, counterfactual} - set(df.columns):
        raise ValueError("Observed and/or Counterfactual columns are missing in the input dataframe")
    
    # Correction_factor name should be a constant?
    df['correction_factor'] = df[observed]/ df[counterfactual]
    
    return df


def correct_treatment(df : pd.DataFrame, treatment_observed : str = 't_obs', treatment_cf : str ='t_cf', cg_observed : str ='c_obs',cg_counterfactual : str ='c_cf'):
    """
        Given a dataframe compute the corrected counterfactual

    Parameters
    ----------
    df : a dataframe with treatment and comparison group observed and counterfactuals

    Returns
    -------
    Dataframe with counterfactual corrected
    """

    df_corrected = pct_correction_factor(df, cg_observed, cg_counterfactual)

    if treatment_cf not in df.columns:
        raise ValueError("Treatment Counterfactual column missing in the input dataframe")
        
    df_corrected['t_cf_corrected'] = df_corrected['correction_factor'] * df[treatment_cf]

    return df_corrected