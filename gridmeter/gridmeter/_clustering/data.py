import pandas as pd


def set_df_index(df:pd.DataFrame):
    """
    sets index of dataframe they are in expected format
    """
    if df.index.names == ["id", "time"]:
         return df
    
    return df.set_index(["id", "time"])