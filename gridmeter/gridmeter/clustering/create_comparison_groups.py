"""
main module to import and to expose external api
"""

from __future__ import annotations

import pandas as pd

from gridmeter.clustering import settings as _settings, cluster as _cluster


class Clustering:
    def __init__(self, settings: _settings.Settings | None):
        if settings is None:
            settings = _settings.Settings()

        self.settings = settings
        self.cg_match = None

    def get_comparison_group(self, df_ls_t: pd.DataFrame, df_ls_cp: pd.DataFrame):
        cg_match = _cluster.ClusterResult.from_comparison_pool_loadshapes_and_settings(
            df_cp_ls=df_ls_cp, s=self.settings
        )
        self.cg_match = cg_match

        treatment_weight_df = cg_match.get_match_treatment_to_cluster_df(
            treatment_loadshape_df=df_ls_t
        )
        return cg_match.cluster_df, treatment_weight_df
