import numpy as np
import pandas as pd

from gridmeter.individual_meter_matching.settings import Settings
from gridmeter.individual_meter_matching.distance_calc_selection import DistanceMatching


# TODO store closest matches to each treatment meter
# TODO add weights as column in df_cg in case of duplicates/weights
class Individual_Meter_Matching:
    def __init__(self, settings: Settings | None):
        if settings is None:
            self.settings = Settings()
        else:
            self.settings = settings

        self.dist_metric = settings.distance_metric
        if self.dist_metric == "manhattan":
            self.dist_metric = "cityblock"

    def get_comparison_group(self, df_ls_t, df_ls_cp, weights=None):
        distance_matching = DistanceMatching(self.settings)

        df_cg = distance_matching.get_comparison_group(
            df_ls_t, df_ls_cp, weights=weights
        )
        df_cg["cluster"] = 1
        df_cg = df_cg.reset_index().set_index("id")

        # Create df_t_coeffs
        t_ids = df_ls_t.index.unique()
        coeffs = np.ones(t_ids.values.size)

        df_t_coeffs = pd.DataFrame(coeffs, index=t_ids, columns=["pct_cluster_1"])
        df_t_coeffs.index.name = "id"

        return df_cg, df_t_coeffs
