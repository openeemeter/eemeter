#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

   Copyright 2020 GRIDmeter™ contributors

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

import numpy as np
import pandas as pd
import scipy.spatial
import scipy

__all__ = ("DistanceMatching",)


class DistanceMatching:
    """
    Parameters
    ----------
    treatment_group: pd.DataFrame
        A dataframe representing treatment group meters, indexed by id, with each column being a data point in a usage pattern.
    comparison_pool: pd.DataFrame
        A dataframe representing comparison pool meters, indexed by id, with each column being a data point in a usage pattern.
    weights: list
        A list of floats (must be of length of the treatment group columns) to scale the usage patterns in order to ensure that certain components of usage have higher weights towards matching than others.
    n_treatments_per_chunk: int
        Due to local memory limitations, treatment meters can be chunked so that the cdist calculation can happen in memory. 10,000 meters appear to be sufficient for most memory constraints.
    """

    def __init__(
        self,
        treatment_group,
        comparison_pool,
        weights=None,
        n_treatments_per_chunk=10000,
    ):
        self.n_treatments_per_chunk = n_treatments_per_chunk

        self.weights = weights
        self.treatment_group = treatment_group
        self.comparison_pool = comparison_pool
        if weights:
            self.treatment_group = self.treatment_group * self.weights
            self.comparison_pool = self.comparison_pool * self.weights

        self.treatment_group_chunks = [
            self.treatment_group[
                chunk * n_treatments_per_chunk : (chunk + 1) * n_treatments_per_chunk
            ]
            for chunk in range(int(len(treatment_group) / n_treatments_per_chunk) + 1)
        ]

    def _get_min_distance_from_matrix_df(self, dist_df):
        match_cols = dist_df.columns[(np.argmin(dist_df.values, axis=1))]
        dist = np.diag(dist_df[match_cols])
        return pd.DataFrame({"closest": match_cols, "dist": dist})

    def _get_next_best_matches(self, treatment_distances_df, best_match):
        # The purpose of this for loop is to attempt to find the 'next best match'
        # for treatment meters matched to a comparison pool meter that has already
        # had a previous match
        # get the matched unduplicated meters
        treatment_match_key = best_match.i_key
        comparison_match_key = best_match.closest

        # Get a matrix that only contains the treatments
        # that were matched to an already matched comparison pool meter
        # and therefore need a 'next-best' match
        # keep only the ones that don't have a best match yet
        treatment_distances_unmatched_df = treatment_distances_df.loc[
            ~np.isin(treatment_distances_df.index, treatment_match_key)
        ].copy()
        # drop the columns of matched comparison meters so that you can't get duplicates
        treatment_distances_unmatched_df = treatment_distances_unmatched_df.drop(
            comparison_match_key, axis=1
        )

        if treatment_distances_unmatched_df.columns.empty:
            # There are no unmatched comparison pool meters remaining
            return best_match, False

        # Next-best match is found for the unmatched treatment meters
        treatment_matches_next_best = self._get_min_distance_from_matrix_df(
            treatment_distances_unmatched_df
        )
        # must set the index back to the right values
        treatment_matches_next_best.index = treatment_distances_unmatched_df.index

        # Get the new best matches
        tm = treatment_matches_next_best.reset_index().rename(
            {"index": "i_key"}, axis=1
        )
        # grab the indexes of the best matches
        next_best_match = tm.groupby("closest").apply(lambda x: x.loc[x.dist.idxmin()])
        # append the next best matches
        best_match = best_match.append(next_best_match)

        # Check if there are any unmatched treatment meters and stop if there are none
        if len(treatment_distances_df) == len(best_match):
            return best_match, False
        return best_match, True

    def _get_best_match(self, treatment_distances_df, n_max_duplicate_check_rounds):
        """
        Parameters
        ----------
        treatment_distances_df: pd.DataFrame
            A matrix where the row indices (i) are treatment meters, the columns (j) are
            comparison pool meters, and the values are the calculated distance between
            treatment[i] and comparison_pool[j]
        n_max_duplicate_check_rounds: int
            The number of rounds of checking for 'next best matches' if multiple treatment meters matched to the same comparison group meters. This number dictates how many iterations of 'next best matching' will take place.
        """

        treatment_matches = self._get_min_distance_from_matrix_df(
            treatment_distances_df
        )
        # reset the index to match on distance for duplicated vals
        tm = treatment_matches.reset_index().rename({"index": "i_key"}, axis=1)
        # grab the indexes of the best matches
        best_match = tm.groupby("closest").apply(lambda x: x.loc[x.dist.idxmin()])

        for run_i in range(0, n_max_duplicate_check_rounds):
            (best_match, check_again) = self._get_next_best_matches(
                treatment_distances_df, best_match
            )
            if not check_again:
                break
        # put back in any unmatched with labels
        tm = treatment_matches.loc[~np.isin(treatment_matches.index, best_match.i_key)]
        bm = best_match.rename(
            {"closest": "match", "dist": "distance", "i_key": "index"}, axis=1
        ).set_index("index")
        bm["duplicated"] = False
        tm = tm.rename({"closest": "match", "dist": "distance"}, axis=1)
        tm["duplicated"] = True
        treatment_matches_df = bm.append(tm).sort_index()
        return treatment_matches_df

    def get_comparison_group(
        self,
        n_matches_per_treatment,
        metric="euclidean",
        max_distance_threshold=None,
        n_max_duplicate_check_rounds=10,
    ):
        """
        Parameters
        ----------
        n_matches_per_treatment: int
            number of comparison matches desired per treatment
        metric: str or callable
            A string or callable that goes into numpy's cdist function
        max_distance_threshold: int
            The maximum distance that a comparison group match can have with a given
            treatment meter. These meters are filtered out after all matching has completed.
        n_max_duplicate_check_rounds: int
            The number of rounds of checking for 'next best matches' if multiple treatment meters matched to the same comparison group meters. This number dictates how many iterations of 'next best matching' will take place.
        """
        # chunk the treatment group due to memory constraints

        # for each chunk, for each of n_matches, compose a comparison group
        comparison_group = pd.DataFrame(columns=["match", "distance", "duplicated"])
        for treatment_group_chunk in self.treatment_group_chunks:
            mat = scipy.spatial.distance.cdist(
                treatment_group_chunk.values,
                self.comparison_pool.values,
                metric=metric,
            )
            dist_df = pd.DataFrame(mat)
            # get the best n matches
            for n in range(n_matches_per_treatment):
                dist_df = dist_df[
                    dist_df.columns[~dist_df.columns.isin(comparison_group["match"])]
                ]
                if dist_df.empty:
                    continue
                new_df = self._get_best_match(dist_df, n_max_duplicate_check_rounds)
                comparison_group = comparison_group.append(new_df)

        # rename columns and reindex to get original ids back
        comparison_group = comparison_group.reset_index().rename(
            {"index": "treatment"}, axis=1
        )
        comparison_group.index = self.comparison_pool.iloc[
            comparison_group["match"]
        ].index
        comparison_group["treatment"] = self.treatment_group.iloc[
            comparison_group["treatment"]
        ].index
        comparison_group.drop("match", axis=1, inplace=True)
        comparison_group.index.name = "id"

        return (
            comparison_group[comparison_group["distance"] < max_distance_threshold]
            if max_distance_threshold
            else comparison_group
        )
