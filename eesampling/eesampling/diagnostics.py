#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

   Copyright 2020 EESampling contributors

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine
from plotnine import *
from itertools import combinations
from scipy.stats import ttest_ind, ks_2samp
from scipy.spatial.distance import pdist
from scipy.stats import chisquare

from .bins import Binning, BinnedData, sample_bins


def concat_dfs(dfs_to_concat, concat_col, concat_values):
    if len(dfs_to_concat) != len(concat_values):
        raise ValueError("dfs_to_concat should be the same length as concat_values")
    return pd.concat(
        [
            df.assign(**{concat_col: value})
            for df, value in zip(dfs_to_concat, concat_values)
        ],
        sort=False,
    )


def t_and_ks_test(x, y, thresh=0.05):
    t_p = "{:,.3f}".format(ttest_ind(x, y).pvalue)
    ks_p = "{:,.3f}".format(ks_2samp(x, y).pvalue)

    t_p = ttest_ind(x, y).pvalue
    ks_p = ks_2samp(x, y).pvalue
    t_ok = t_p > thresh
    ks_ok = ks_p > thresh
    t_p_str = "t pval: {:,.3f}".format(t_p)
    ks_p_str = "KS pval: {:,.3f}".format(ks_p)

    return pd.Series(
        {
            "ks_ok": ks_ok,
            "t_ok": t_ok,
            "ks_p": ks_p_str,
            "t_p": t_p_str,
            "t_value": t_p,
            "ks_value": ks_p,
        }
    )


class DiagnosticPlotter:
    def quantile(self, df, df_equiv, cols=None):
        if cols is None:
            cols = self.default_cols

        df_quantile = df[["population"] + cols].melt(id_vars=["population"])
        quantile_range = np.arange(0.005, 1.0, 0.01)
        df_quantile = (
            df_quantile.groupby(["population", "variable"])
            .apply(
                lambda x: pd.DataFrame(
                    {
                        "quantile": quantile_range,
                        "value": x["value"].quantile(quantile_range),
                    }
                )
            )
            .reset_index()
        )

        plotnine.options.figure_size = (6, 3 * df_quantile.variable.nunique())

        base_plot = (
            ggplot(df_quantile, aes(x="quantile", y="value", color="population"))
            + geom_point()
            + facet_wrap("~variable", scales="free_y", ncol=1)
            + theme_bw()
        )

        df_range = (
            df_quantile.groupby("variable")
            .apply(lambda df: pd.Series({"min": df.value.min(), "max": df.value.max()}))
            .reset_index()
        )

        df_equiv = df_equiv.merge(df_range)
        df_equiv["x"] = 0
        df_equiv["y"] = (df_equiv["max"] - df_equiv["min"]) * 0.95 + df_equiv["min"]
        df_equiv["y2"] = (df_equiv["max"] - df_equiv["min"]) * 0.8 + df_equiv["min"]

        p = (
            base_plot
            + geom_label(
                aes(label="t_p", fill="t_ok", x="x", y="y"),
                data=df_equiv,
                ha="left",
                va="top",
                color="black",
                size=10,
            )
            + geom_label(
                aes(label="ks_p", fill="ks_ok", x="x", y="y2"),
                data=df_equiv,
                ha="left",
                va="top",
                color="black",
                size=10,
            )
            + scale_fill_manual({True: "lightgreen", False: "orange"}, guide=None)
            + scale_color_discrete()
        )

        return p

    def scatter(self, df, cols=None):
        if cols is None:
            cols = self.default_cols
        col_pairs = combinations(cols, 2)
        plots = [self._scatter(df, p[0], p[1]) for p in col_pairs]
        return [p for p in plots]

    def _scatter(self, df, col_x, col_y):
        def sample_if_too_big(df):
            if len(df) > 2000:
                df = df.sample(2000)
            return df

        df = (
            df.groupby("population", group_keys=False)
            .apply(sample_if_too_big)
            .reset_index()
        )

        plotnine.options.figure_size = (12, 5)
        base_plot = (
            ggplot(df, aes(x=col_x, y=col_y, color="population"))
            + geom_point()
            + facet_wrap("~population", nrow=1)
            + theme_bw()
        )
        outlier_bins = self.data_treatment.outlier_bins
        outlier_bins = outlier_bins[outlier_bins["outlier"]]["bin"].values
        df_rects = self.binning.edges_xy(col_x, col_y)
        df_rects = df_rects[~df_rects["bin"].isin(outlier_bins)]

        # due to plotnine bug
        df_rects[col_x] = 0
        df_rects[col_y] = 0
        p = base_plot + geom_rect(
            aes(xmin="x_min", xmax="x_max", ymin="y_min", ymax="y_max"),
            data=df_rects,
            color="black",
            fill=None,
            size=0.2,
        )

        return p

    def histogram(self, df, cols=None):
        if cols is None:
            cols = self.default_cols
        return [self._histogram(df, c) for c in cols]

    def _histogram(self, df, col):

        plotnine.options.figure_size = (12, 5)
        p = (
            ggplot(df, aes(x=col, fill="population"))
            + geom_histogram(bins=30)
            + facet_wrap("~population", nrow=1, scales="free_y")
            + theme_bw()
        )

        outlier_bins = self.data_treatment.outlier_bins
        outlier_bins = outlier_bins[outlier_bins["outlier"]]["bin"].values
        df_rects = self.binning.edges_xy(col, col)
        df_rects = df_rects[~df_rects["bin"].isin(outlier_bins)]

        # due to plotnine bug
        df_rects[col] = 0
        p = p + geom_rect(
            aes(xmin="x_min", xmax="x_max", ymin=-np.inf, ymax=np.inf),
            data=df_rects,
            color="black",
            fill=None,
            size=0.2,
        )

        return p


class Diagnostics(DiagnosticPlotter):
    def __init__(self, model):
        self.model = model
        self.binning = self.model.binning
        self.data_treatment = self.model.data_treatment
        self.data_pool = self.model.data_pool
        self.sampled = self.model.sampled

        self.treatment_label = self.model.treatment_label
        self.pool_label = self.model.pool_label
        self.data_sample = self.model.data_sample

        # these two will always exist
        df_treatment = self.model.data_treatment.df
        df_pool = self.model.data_pool.df
        self.default_cols = self.model.col_names

        self.available_equiv_labels = [
            self.treatment_label,
            self.pool_label,
            "sample",
        ]
        df_sample = (
            self.data_sample.df
            if self.data_sample is not None
            else pd.DataFrame()
        )
        self.labeled_dfs = [df_treatment, df_pool, df_sample]

        self.df_all = concat_dfs(
            self.labeled_dfs, "population", self.available_equiv_labels
        )

    def histogram(self, cols=None):
        return super().histogram(self.df_all, cols)

    def scatter(self, cols=None):
        return super().scatter(self.df_all, cols)

    def quantile_equivalence(self, cols=None):
        df_equiv = self.equivalence(cols)
        return super().quantile(self.df_all, df_equiv, cols=cols)

    def _check_equiv_labels(self, equiv_label_x, equiv_label_y):
        if (
            equiv_label_x is not None
            and equiv_label_x not in self.available_equiv_labels
        ):
            raise ValueError(
                f"equiv_label_x must be one of: {self.available_equiv_labels}"
            )
        if (
            equiv_label_y is not None
            and equiv_label_y not in self.available_equiv_labels
        ):
            raise ValueError(
                f"equiv_label_y must be one of: {self.available_equiv_labels}"
            )
        equiv_label_x = equiv_label_x if equiv_label_x else self.treatment_label
        equiv_label_y = (
            equiv_label_y
            if equiv_label_y
            else ("sample" if self.data_sample else self.pool_label)
        )
        return equiv_label_x, equiv_label_y

    def records_based_equivalence(
        self,
        df_for_equivalence,
        groupby_col,
        value_col,
        how,
        id_col="id",
        equiv_label_x=None,
        equiv_label_y=None,
        *args,
        **kwargs,
    ):
        if how == "euclidean":
            return self.records_based_equivalence_euclidean(
                df_for_equivalence=df_for_equivalence,
                groupby_col=groupby_col,
                value_col=value_col,
                id_col=id_col,
                equiv_label_x=equiv_label_x,
                equiv_label_y=equiv_label_y,
            )
        elif how == "chisquare":
            return self.records_based_equivalence_chisquare(
                df_for_equivalence=df_for_equivalence,
                groupby_col=groupby_col,
                value_col=value_col,
                id_col=id_col,
                equiv_label_x=equiv_label_x,
                equiv_label_y=equiv_label_y,
                *args,
                **kwargs,
            )
        raise ValueError("how must be one of: ['euclidean', 'chisquare']")

    def records_based_equivalence_euclidean(
        self,
        df_for_equivalence,
        groupby_col,
        value_col,
        id_col="id",
        equiv_label_x=None,
        equiv_label_y=None,
    ):
        """
        Attributes
        ----------
        equiv_label_x: str
            First label to measure equivalence against (defaults to treatment label) 
        equiv_label_y: str
            Second label to measure equivalence against (defaults to sample if available,
             otherwise defaults to full pool set)
        """
        equiv_label_x, equiv_label_y = self._check_equiv_labels(
            equiv_label_x, equiv_label_y
        )

        df = self.df_all[["population", "id"]].copy()
        df_combined = df.set_index("id", drop=True).join(
            df_for_equivalence.set_index(id_col, drop=True)
        )
        equiv_x = (
            df_combined[df_combined["population"] == equiv_label_x]
            .groupby(groupby_col)[value_col]
            .mean()
        )
        equiv_y = (
            df_combined[df_combined["population"] == equiv_label_y]
            .groupby(groupby_col)[value_col]
            .mean()
        )
        return equiv_x.to_frame(), equiv_y.to_frame(), pdist([equiv_x, equiv_y])[0]

    def records_based_equivalence_chisquare(
        self,
        df_for_equivalence,
        groupby_col,
        value_col,
        id_col="id",
        equiv_label_x=None,
        equiv_label_y=None,
        chisquare_n_values_per_bin=25,
        chisquare_is_fixed_width=False,
    ):
        """
        Attributes
        ----------
        equiv_label_x: str
            First label to measure equivalence against (defaults to treatment label) 
        equiv_label_y: str
            Second label to measure equivalence against (defaults to sample if available,
             otherwise defaults to full pool set)
        """
        equiv_label_x, equiv_label_y = self._check_equiv_labels(
            equiv_label_x, equiv_label_y
        )

        df = self.df_all[["population", "id"]].copy()
        df_combined = df.set_index("id", drop=True).join(
            df_for_equivalence.set_index(id_col, drop=True)
        )

        chisquare_stats = []
        equiv_x = []
        equiv_y = []
        for groupby_col_num in df_combined[groupby_col].unique():
            df_equiv_x = df_combined[
                (df_combined["population"] == equiv_label_x)
                & (df_combined[groupby_col] == groupby_col_num)
            ]
            df_equiv_y = df_combined[
                (df_combined["population"] == equiv_label_y)
                & (df_combined[groupby_col] == groupby_col_num)
            ]
            # Select num_bins from number of treatments / chosen ratio
            num_bins = max(int(len(df_equiv_x) / chisquare_n_values_per_bin), 1)

            binning_x = Binning()
            binning_x.bin(
                df_equiv_x[value_col],
                value_col,
                num_bins,
                fixed_width=chisquare_is_fixed_width,
            )

            binned_data_x = BinnedData(
                df_equiv_x[[value_col]],
                binning_x,
                min_n_treatment_per_bin=chisquare_n_values_per_bin,
            )

            binning_y = Binning()
            binning_y.bin(
                df_equiv_y[value_col],
                value_col,
                num_bins,
                fixed_width=chisquare_is_fixed_width,
            )

            binned_data_y = BinnedData(
                df_equiv_y[[value_col]],
                binning_y,
                min_n_treatment_per_bin=chisquare_n_values_per_bin,
            )
            chisquare_x = binned_data_x.df.groupby("_bin_label")[value_col].mean()
            chisquare_y = binned_data_y.df.groupby("_bin_label")[value_col].mean()
            chisquare_stats.append(chisquare(chisquare_x, chisquare_y).statistic)

            chisquare_x_df = chisquare_x.to_frame()
            chisquare_x_df[groupby_col] = groupby_col_num
            chisquare_y_df = chisquare_y.to_frame()
            chisquare_y_df[groupby_col] = groupby_col_num

            equiv_x.append(chisquare_x_df)
            equiv_y.append(chisquare_y_df)
        return (
            pd.concat(equiv_x).reset_index(),
            pd.concat(equiv_y).reset_index(),
            sum(chisquare_stats) if chisquare_stats else np.inf,
        )

    def equivalence(self, cols=None, equiv_label_x=None, equiv_label_y=None):
        """
        Attributes
        ----------
        cols: str
            Columns to plot and calculate equivalence for. Defaults to all available cols. 
        equiv_label_x: str
            First label to measure equivalence against (defaults to treatment label) 
        equiv_label_y: str
            Second label to measure equivalence against (defaults to sample if available,
             otherwise defaults to full pool set)
        """
        if (
            equiv_label_x is not None
            and equiv_label_x not in self.available_equiv_labels
        ):
            raise ValueError(
                f"equiv_label_x must be one of: {self.available_equiv_labels}"
            )
        if (
            equiv_label_y is not None
            and equiv_label_y not in self.available_equiv_labels
        ):
            raise ValueError(
                f"equiv_label_y must be one of: {self.available_equiv_labels}"
            )
        equiv_label_x = equiv_label_x if equiv_label_x else self.treatment_label
        equiv_label_y = (
            equiv_label_y
            if equiv_label_y
            else ("sample" if self.data_sample else self.pool_label)
        )
        cols = cols if cols else self.default_cols

        df = self.df_all[["population"] + cols].melt(id_vars=["population"])
        return (
            df.groupby("variable")
            .apply(
                lambda x: t_and_ks_test(
                    x[x["population"] == equiv_label_x].value.dropna(),
                    x[x["population"] == equiv_label_y].value.dropna(),
                )
            )
            .reset_index()
        )

    def equivalence_passed(self, cols=None):
        df = self.equivalence(cols=cols)
        return all(df["ks_ok"]) & all(df["t_ok"])

    def count_bins(self):

        df_treatment = self.data_treatment.count_bins(skip_outliers=True).rename(
            columns={
                "n": f"n_{self.treatment_label}",
                "n_pct": f"n_pct_{self.treatment_label}",
            }
        )

        df_pool = self.data_pool.count_bins(skip_outliers=False).rename(
            columns={
                "n": f"n_{self.pool_label}",
                "n_pct": f"n_pct_{self.pool_label}",
            }
        )

        df = df_treatment.merge(df_pool)

        if self.sampled:
            df_sample = self.data_sample.count_bins(skip_outliers=False).rename(
                columns={"n": f"n_sampled", "n_pct": f"n_pct_sampled"}
            )
            df = df.merge(df_sample)
        return df

    def n_sampled_to_n_treatment_ratio(self):
        bin_df = self.count_bins()
        if bin_df.empty:
            return 0
        else:
            return (
                (bin_df["n_sampled"] / bin_df[f"n_{self.treatment_label}"])
                .min()
                .astype(int)
            )
