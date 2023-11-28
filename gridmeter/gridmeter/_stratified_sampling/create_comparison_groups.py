from __future__ import annotations

import numpy as np
import pandas as pd

from gridmeter._utils.base_comparison_group import Comparison_Group_Algorithm

from gridmeter._stratified_sampling.model import StratifiedSampling
from gridmeter._stratified_sampling.bins import ModelSamplingException
from gridmeter._stratified_sampling.diagnostics import StratifiedSamplingDiagnostics
from gridmeter._stratified_sampling.bin_selection import StratifiedSamplingBinSelector

from gridmeter._stratified_sampling.settings import Settings


class Stratified_Sampling(Comparison_Group_Algorithm):
    def __init__(self, settings: Settings | None = None):
        if settings is None:
            settings = Settings()

        self.settings = settings
        self.df_raw = None

        self.model = StratifiedSampling()

        for settings in self.settings.STRATIFICATION_COLUMN:
            self.model.add_column(
                settings.COLUMN_NAME,
                n_bins=settings.N_BINS,
                min_value_allowed=settings.MIN_VALUE_ALLOWED,
                max_value_allowed=settings.MAX_VALUE_ALLOWED,
                fixed_width=settings.IS_FIXED_WIDTH,
                auto_bin_require_equivalence=settings.AUTO_BIN_EQUIVALENCE,
            )


    def _create_clusters_df(self, ids):
        clusters = pd.DataFrame(ids, columns=["id"])
        clusters["cluster"] = 0
        clusters["weight"] = 1.0

        clusters = clusters.reset_index().set_index("id")
        clusters = clusters[["cluster", "weight"]]

        return clusters


    def _create_treatment_weights_df(self, ids):
        coeffs = np.ones(ids.size)

        treatment_weights = pd.DataFrame(coeffs, index=ids, columns=["pct_cluster_0"])
        treatment_weights.index.name = "id"

        return treatment_weights
    
    def _create_output_dfs(self, t_ids):
        self.df_raw = self.model.data_sample.df

        # Create comparison group
        df_cg = self.df_raw[self.df_raw["_outlier_bin"] == False]
        clusters = self._create_clusters_df(df_cg["meter_id"].unique())

        # Create treatment_weights
        
        treatment_weights = self._create_treatment_weights_df(t_ids)

        # Assign dfs to self
        self.clusters = clusters
        self.treatment_weights = treatment_weights

        return clusters, treatment_weights


    def get_comparison_group(self, df_treatment, df_comparison_pool):
        settings = self.settings
        self.model.fit_and_sample(
            df_treatment, 
            df_comparison_pool,
            n_samples_approx=settings.N_SAMPLES_APPROX,
            relax_n_samples_approx_constraint=settings.RELAX_N_SAMPLES_APPROX_CONSTRAINT,
            min_n_treatment_per_bin=settings.MIN_N_TREATMENT_PER_BIN,
            min_n_sampled_to_n_treatment_ratio=settings.MIN_N_SAMPLED_TO_N_TREATMENT_RATIO,
            random_seed=settings.SEED,
        )

        t_ids = df_treatment["meter_id"].unique()
        clusters, treatment_weights = self._create_output_dfs(t_ids)

        return clusters, treatment_weights


    def diagnostics(self):
        if self.df_raw is None:
            raise RuntimeError("Must run get_comparison_group() before calling diagnostics()")
        
        return StratifiedSamplingDiagnostics(model=self.model)
    

class Distance_Stratified_Sampling(Stratified_Sampling):
    def __init__(self, settings: Settings | None = None):
        if settings is None:
            settings = Settings()

        super().__init__(settings=settings)


    def _split_df(self, df):
        feature_cols = [col for col in df.columns if not isinstance(col, int)]
        loadshape_cols = [col for col in df.columns if isinstance(col, int)]

        return df[feature_cols], df[loadshape_cols]


    def get_comparison_group(self, df_treatment, df_comparison_pool):
        settings = self.settings

        # split dfs into features and loadshapes
        df_features_t, df_ls_t = self._split_df(df_treatment)
        df_features_cp, df_ls_cp = self._split_df(df_comparison_pool)

        # combine loadshapes into one df
        df_ls = pd.concat([df_ls_t, df_ls_cp])

        self.model_bin_selector = StratifiedSamplingBinSelector(
            self.model,
            df_features_t,
            df_features_cp,
            equivalence_feature_ids=df_ls.index,
            equivalence_feature_matrix=df_ls,
            equivalence_method="chisquare",
            df_id_col = "meter_id",
            min_n_bins=2,
            max_n_bins=8,
            random_seed=settings.SEED,
        )

        t_ids = df_treatment["meter_id"].unique()
        clusters, treatment_weights = self._create_output_dfs(t_ids)

        return clusters, treatment_weights