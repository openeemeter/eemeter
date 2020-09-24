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

import copy
import itertools
import logging
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


class StratifiedSamplingBinSelector(object):
    def __init__(
        self,
        model,
        df_treatment,
        df_pool,
        df_for_equivalence,
        equivalence_groupby_col,
        equivalence_value_col,
        equivalence_id_col,
        how,
        n_samples_approx=5000,
        min_n_treatment_per_bin=0,
        random_seed=1,
        min_n_sampled_to_n_treatment_ratio=0.25,
        min_n_bins=1,
        max_n_bins=8,
        chisquare_n_values_per_bin=25,
        chisquare_is_fixed_width=False,
        relax_n_samples_approx_constraint=True,
    ):
        """
        Finds an optimal stratified sampling bin configuration which minimizes
        distance between treatmnt and comparison groups.  A bin configuration
        is a number of bins, `n_c`, for each stratification column `c`, where 
        c is an integer between `min_n_bins` and `max_n_bin` inclusive.  
        Using a grid search, all possible bin configurations will be constructed and
        tested, and the confiuration which minimizes treatment-comparison distance
        will be returned.  Distance is measured on a set of features 
        provided in `df_for_equivalence` in 'long' format, i.e. multiple rows
        per meter, one column holds feature name, one column holds feature value.

        Distance may be computed in one of two ways, according to the `how` parameter:

        `euclidean`: Compute the mean value of each feature for treatment and comparison, 
        and compute Euclidean distance of the means.

        `chisquare`: For each feature, first cut into quantiles and compute the mean
        value per quantile for treatment and comparison. Then compute the
        chisquare statistic of the means for each feature, and sum the statistic
        across all features.

        Example usage:

            m = StratifiedSampling()
            m.add_column('annual_usage', min_value=0, max_value=20000)
            m.add_column('summer_usage', min_value=0, max_value=1000)
            s = StratifiedSamplingBinSelector(m, df_treatment, df_pool,
                df_for_equivalence, 'month', 'value', 'id', 'chisquare')
            results = s.results_as_json()
            df_comparison = m.data_sample.df


        Attributes
        ==========
        model: eesampling.StratifiedSampling
            Model with stratification columns added.
        df_treatment: pandas.DataFrame
            dataframe to use for constructing the stratified sampling bins.
        df_pool: pandas.DataFrame
            dataframe to sample from according to the constructed stratified sampling bins.
        df_for_equivalence: pandas.DataFrame
            dataframe with featues to use for computing equivalence, in 'long' form
        equivalence_groupby_col: str
            Name of column in `df_for_equivalence` which contains the feature name
        equivalence_value_col: str
            Name of column in `df_for_equivalence` which contains the feature value
        equivalence_id_col: str
            Name of column in `df_for_equivalence` which contains the ID.  `df_treatment`
            and `df_pool` must have an ID column with the same name.
        how: str
            Method for computing distance -- either 'euclidean' or 'chisquare'.
        n_samples_aprox: int
            approximate number of total samples from df_pool which are used to construct
            the comparison group. It is approximate because
            there may be some slight discrepencies around the total count to ensure
            that each bin has the correct percentage of the total.
            A None value means that it will take as many samples as it has available.
        min_n_treatment_per_bin: int
            Minimum number of treatment samples that must exist in a given bin for 
            it to be considered a non-outlier bin (only applicable if there are 
            cols with fixed_width=True)
        min_n_sampled_to_n_treatment_ratio: int
            Minimum number samples that must exist in each bin per treatment datapoint in that bin.
        min_n_bins: int
            Minimum number of bins to use in stratified sampling.
        max_n_bins: int
            Maximum number of bins to use in stratified sampling.
        chisquare_n_values_per_bin: int
            When computing equivalence using `chisquare` method, number of samples per bin.
        chisquare_is_fixed_width: boolean
            When computing equivalence using `chisquare` method, whether to use fixed-width
            bins, or fixed-quantity bins.
        relax_n_samples_approx_constraint: bool
            If True, treats n_samples_approx as an upper bound, but gets as many comparison group
            meters as available up to n_samples_approx. If False, it raises an exception
            if there are not enough comparison pool meters to reach n_samples_approx.
        """
        # Settings
        self.how = how
        self.n_samples_approx = n_samples_approx
        self.min_n_treatment_per_bin = min_n_treatment_per_bin
        self.random_seed = random_seed
        self.min_n_sampled_to_n_treatment_ratio = min_n_sampled_to_n_treatment_ratio
        self.min_n_bins = min_n_bins
        self.max_n_bins = max_n_bins
        self.chisquare_n_values_per_bin = chisquare_n_values_per_bin
        self.chisquare_is_fixed_width = chisquare_is_fixed_width
        self.equivalence_groupby_col = equivalence_groupby_col
        self.equivalence_value_col = equivalence_value_col
        self.equivalence_id_col = equivalence_id_col

        self.model = model
        self.df_treatment = df_treatment
        self.df_pool = df_pool
        self.df_for_equivalence = df_for_equivalence
        self.n_bin_options_df = None
        self.equiv_treatment = None
        self.equiv_samples = []

        if len(self.model.columns) == 0:
            raise ValueError("You must add at least one column before fitting.")
        if any([not col["auto_bin"] for name, col in self.model.columns.items()]):
            raise ValueError("This form of fitting only works n_bins is not set")
        logger.debug(self.model.columns)
        min_distance = float("Inf")
        min_columns = None

        column_names = list(self.model.columns.keys())
        n_bin_results = []
        self.n_bin_options_df = pd.DataFrame(
            [
                {column_names[i - 1]: c for i, c in enumerate(comb)}
                for comb in itertools.product(
                    range(min_n_bins, max_n_bins + 1), repeat=len(column_names)
                )
            ]
        )
        disqualified_n_bin_options = []
        for n_bin_option in self.n_bin_options_df.to_dict("records"):
            [
                self.model.set_n_bins(name, n_bins)
                for name, n_bins in n_bin_option.items()
            ]
            bins_selected_str = self.model.get_all_n_bins_as_str()

            if n_bin_option in disqualified_n_bin_options:
                logger.debug(f"Skipping {bins_selected_str} (disqualified)")
                continue
            self.model.fit(
                self.df_treatment,
                min_n_treatment_per_bin=min_n_treatment_per_bin,
                random_seed=random_seed,
            )

            self.model.sample(
                self.df_pool,
                n_samples_approx=n_samples_approx,
                random_seed=random_seed,
                relax_n_samples_approx_constraint=relax_n_samples_approx_constraint,
            )
            n_sampled_to_n_treatment_ratio = (
                self.model.diagnostics().n_sampled_to_n_treatment_ratio()
            )
            if (
                not self.model.relax_ratio_constraint
                and n_sampled_to_n_treatment_ratio < min_n_sampled_to_n_treatment_ratio
            ):
                logger.info(
                    f"Insufficient pool data for {bins_selected_str}:"
                    f"found {n_sampled_to_n_treatment_ratio}:1 but need "
                    f"{min_n_sampled_to_n_treatment_ratio}:1."
                )
                disqualified_options = self.n_bin_options_df.loc[
                    (
                        self.n_bin_options_df[list(n_bin_option)]
                        >= pd.Series(n_bin_option)
                    ).all(axis=1)
                ].to_dict("records")
                disqualified_n_bin_options.extend(disqualified_options)
                n_bin_results.append(
                    dict(
                        **n_bin_option,
                        **{
                            "distance": None,
                            "status": "FAILED",
                            "bins_selected_str": bins_selected_str,
                        },
                    )
                )
                continue
            equiv_treatment, equiv_sample, equivalence_distance = self.model.diagnostics().records_based_equivalence(
                self.df_for_equivalence,
                equivalence_groupby_col,
                equivalence_value_col,
                id_col=equivalence_id_col,
                how=how,
                chisquare_n_values_per_bin=chisquare_n_values_per_bin,
                chisquare_is_fixed_width=chisquare_is_fixed_width,
            )
            n_bin_results.append(
                dict(
                    **n_bin_option,
                    **{
                        "distance": equivalence_distance,
                        "status": "SUCCEEDED",
                        "bins_selected_str": bins_selected_str,
                    },
                )
            )
            # build a dataframe with the equivalence vectors so we can plot them
            equiv_sample["bin_str"] = bins_selected_str
            self.equiv_samples.append(equiv_sample.copy(deep=True))

            logging.info(
                f"Computing bins: {bins_selected_str} distance: "
                f"{equivalence_distance:.2f}, "
                f"pct: {100*equivalence_distance/sum(equiv_treatment[equivalence_value_col]):.2f}"
            )
            if equivalence_distance < min_distance:
                min_distance = equivalence_distance
                min_columns = copy.deepcopy(self.model.columns)
        self.n_bin_results = pd.DataFrame(n_bin_results)
        if not min_columns:
            raise ValueError("No valid bin configurations were discovered")

        # same for all of them anyway
        # TODO (ssuffian): Calculate this cleaner
        equiv_treatment.name = self.model.treatment_label
        self.equiv_treatment = equiv_treatment

        self.model.columns = min_columns
        bins_selected_str = self.model.get_all_n_bins_as_str()
        logging.info(
            f"Selected bin: {bins_selected_str} distance: "
            f"{min_distance:.2f}, "
            f"pct: {100*min_distance/sum(equiv_treatment[equivalence_value_col]):.2f}, "
            f"random_seed: {random_seed}"
        )
        self.model.fit(
            self.df_treatment,
            min_n_treatment_per_bin=min_n_treatment_per_bin,
            random_seed=random_seed,
        )
        # if n_samples_approx is None, use the maximum available.
        self.model.sample(
            self.df_pool,
            n_samples_approx=n_samples_approx,
            random_seed=random_seed,
            relax_n_samples_approx_constraint=relax_n_samples_approx_constraint,
        )
        self.n_samples_approx = n_samples_approx

        # get averages that can be accessed later

        def _get_equiv_comparison_pool():
            df_combined = self.df_for_equivalence[
                self.df_for_equivalence[self.equivalence_id_col].isin(
                    self.df_pool["id"]
                )
            ]
            equiv_full_avg = df_combined.groupby(self.equivalence_groupby_col)[
                self.equivalence_value_col
            ].mean()
            equiv_full_avg.name = "comparison pool"
            return equiv_full_avg

        self.equiv_pool_avg = _get_equiv_comparison_pool()

        self.equiv_treatment_avg = self.equiv_treatment.groupby(
            self.equivalence_groupby_col
        ).mean()
        self.equiv_treatment_avg.columns = ["treatment"]

        self.equiv_samples_avg = (
            pd.concat(self.equiv_samples)
            .groupby(["bin_str", equivalence_groupby_col])
            .mean()
            .reset_index()
            .pivot(
                index=equivalence_groupby_col,
                columns="bin_str",
                values=equivalence_value_col,
            )
        )
        self.bins_selected_str = self.model.get_all_n_bins_as_str()

    def kwargs_as_json(self):
        return {
            "how": self.how,
            "n_samples_approx": self.n_samples_approx,
            "min_n_treatment_per_bin": self.min_n_treatment_per_bin,
            "random_seed": self.random_seed,
            "min_n_sampled_to_n_treatment_ratio": self.min_n_sampled_to_n_treatment_ratio,
            "min_n_bins": self.min_n_bins,
            "max_n_bins": self.max_n_bins,
            "chisquare_n_values_per_bin": self.chisquare_n_values_per_bin,
            "chisquare_is_fixed_width": self.chisquare_is_fixed_width,
            "equivalence_groupby_col": self.equivalence_groupby_col,
            "equivalence_id_col": self.equivalence_id_col,
            "equivalence_value_col": self.equivalence_value_col,
        }

    def results_as_json(self):
        equiv_samples_df = pd.concat(self.equiv_samples)
        selected_sample_df = equiv_samples_df[
            equiv_samples_df["bin_str"] == self.bins_selected_str
        ]

        return {
            "bins_selected": self.bins_selected_str,
            "random_seed": self.random_seed,
            "n_bin_results": self.n_bin_results.to_dict("records"),
            "chisquare_averages": {
                "selected_sample": selected_sample_df.to_dict("records"),
                self.model.treatment_label: self.equiv_treatment.to_dict("records"),
            },
            "averages": {
                "samples": self.equiv_samples_avg.reset_index().to_dict("records"),
                "selected_sample": self.equiv_samples_avg[self.bins_selected_str]
                .reset_index()
                .to_dict("records"),
                self.model.treatment_label: self.equiv_treatment_avg.reset_index().to_dict(
                    "records"
                ),
                self.model.pool_label: self.equiv_pool_avg.reset_index().to_dict(
                    "records"
                ),
            },
        }

    def plot_records_based_equiv_average(self, plot=True):

        equiv_df = pd.concat(
            [self.equiv_treatment_avg, self.equiv_pool_avg, self.equiv_samples_avg],
            axis=1,
        )

        wrong_models = [
            m for m in self.equiv_samples_avg.columns if m != self.bins_selected_str
        ]

        if plot:
            fig, ax = plt.subplots()
            for wm in wrong_models:
                plt.plot(self.equiv_samples_avg[wm], alpha=0.1, color="b")
            equiv_df[[self.bins_selected_str, "treatment", "comparison pool"]].plot(
                color=["k", "r", "k"], style=["-", "-", "."], ax=ax
            )
            plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
