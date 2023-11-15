"""
main module to import and to expose external api
"""

from __future__ import annotations

import pandas as pd

from gridmeter.clustering import settings as _settings, cluster as _cluster, transform as _transform

def master_function(df_t:pd.DataFrame, df_c:pd.DataFrame, settings:_settings.Settings):
    """
    
    """
    
    matcher = _cluster.ClusterResult.from_comparison_pool_loadshapes_and_settings(df_cp_ls=df_c, s=settings)
    treatment_df = matcher.get_match_treatment_to_cluster_df(treatment_loadshape_df=df_t)
    _ = 1
    return matcher, treatment_df


def _main():
    df_ls_t = pd.read_csv("/app/data/df_ls_t.csv")
    df_ls_cp = pd.read_csv("/app/data/df_ls_cp.csv")
    s = _settings.Settings()
    matcher, treatment_df = master_function(df_t=df_ls_t, df_c=df_ls_cp, settings=s)
    print(matcher.cluster_df)
    print(treatment_df)

if __name__ == "__main__":
    _main()