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

from eemeter.gridmeter.individual_meter_matching import highs_settings as _highs_settings
import numpy as np
import pandas as pd

from scipy import sparse
from qpsolvers import solve_ls

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from eemeter.gridmeter.individual_meter_matching.settings import Settings

__all__ = ("DistanceMatching",)



def cp_chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def _distances(ls_t, ls_cp, weights=None, dist_metric="euclidean", n_meters_per_chunk=10000):
    if weights is not None:
        ls_t = ls_t * weights

    # calculate distances in chunks
    n_chunk = len(ls_cp)
    if n_meters_per_chunk < n_chunk:
        n_chunk = n_meters_per_chunk

    dist = []
    for ls_cp_chunk in cp_chunks(ls_cp, n_meters_per_chunk):
        if weights is not None:
            ls_cp_chunk = ls_cp_chunk * weights

        # perform weighted distance calculation
        chunked_dist = cdist(ls_t, ls_cp_chunk, metric=dist_metric)

        dist.append(chunked_dist)

    dist = np.hstack(dist)

    return dist


def highs_fit_comparison_group_loadshape(t_ls, cp_ls, coef_sum=1, solver="highs", settings=None, verbose=False):
    if settings is None:
        if coef_sum == 1:
            settings = _highs_settings.HiGHS_Settings(
                PRIMAL_FEASIBILITY_TOLERANCE=1E-4, 
                DUAL_FEASIBILITY_TOLERANCE=1E-4, 
            )
        else:
            settings = _highs_settings.HiGHS_Settings(
                PRIMAL_FEASIBILITY_TOLERANCE=1, 
                DUAL_FEASIBILITY_TOLERANCE=1, 
            )
        settings = {k.lower(): v for k, v in dict(settings).items()}

    if coef_sum == 1:
        _MIN_X = 1E-6
    else:
        _MIN_X = 5E-3
    
    num_pool_meters = cp_ls.shape[0]

    R = sparse.csc_matrix(cp_ls.T)

    h = np.zeros(num_pool_meters)
    eye = sparse.eye(num_pool_meters, format="csc")
    A = sparse.csc_matrix(np.ones(num_pool_meters))
    b = np.array([coef_sum])

    lb = np.zeros(num_pool_meters)
    ub = np.ones(num_pool_meters)

    x_opt = solve_ls(R, t_ls, G=-eye, h=h, A=A, b=b, lb=lb, ub=ub, solver=solver, verbose=verbose, **settings)

    x_opt[x_opt < 0] = 0
    x_opt[x_opt > 1] = 1
    x_opt[np.abs(x_opt) < _MIN_X] = 0
    x_opt *= coef_sum/x_opt.sum()

    return x_opt


class DistanceMatchingError(Exception):
    pass


class DistanceMatching:
    """
    Parameters
    ----------
    treatment_group: pd.DataFrame
        A dataframe representing treatment group meters, indexed by id, with each column being a data point in a usage pattern.
    comparison_pool: pd.DataFrame
        A dataframe representing comparison pool meters, indexed by id, with each column being a data point in a usage pattern.
    weights: list | 1D np.array
        A list of floats (must be of length of the treatment group columns) to scale the usage patterns in order to ensure that certain components of usage have higher weights towards matching than others.
    n_treatments_per_chunk: int
        Due to local memory limitations, treatment meters can be chunked so that the cdist calculation can happen in memory. 10,000 meters appear to be sufficient for most memory constraints.
    """

    def __init__(
        self,
        settings=None,
    ):
        if settings is None:
            self.settings = Settings()
        elif isinstance(settings, Settings):
            self.settings = settings
        else:
            raise Exception(
                "invalid settings provided to 'individual_metering_matching'"
            )

        self.dist_metric = settings.DISTANCE_METRIC
        if self.dist_metric == "manhattan":
            self.dist_metric = "cityblock"

    def _closest_idx_duplicates_allowed(self, distances, n_match=None):
        if n_match is None:
            n_match = self.settings.N_MATCHES_PER_TREATMENT

        if n_match > distances.shape[1]:
            n_match = distances.shape[1]

        # sort distances by row and get the indices of the sorted distances
        # Note: pypi bottleneck is faster than numpy for this
        cg_idx = np.argpartition(distances, n_match, axis=1)[:, :n_match]

        return cg_idx

    def _closest_idx_duplicates_not_allowed(self, ls_t, ls_cp, distances):
        n_match = self.settings.N_MATCHES_PER_TREATMENT
        selection_method = self.settings.SELECTION_METHOD

        n_treatment = ls_t.shape[0]
        n_pool = ls_cp.shape[0]

        if n_match*n_treatment > n_pool:
            n_match = int(n_pool / n_treatment)

        if n_match == 0:
            raise DistanceMatchingError(f"Not enough treatment pool meters {n_pool} to match with {n_treatment} treatment meters without duplicates")
        
        if selection_method == "minimize_meter_distance":
            # normalize distances by min distance of each row
            # min_dist = np.take_along_axis(distances, self._closest_idx_duplicates_allowed(distances, n_match=1), axis=1)
            # distances = distances / min_dist

            # duplicate rows n_match times
            distances = np.repeat(distances, n_match, axis=0)
            t_idx = np.repeat(np.arange(distances.shape[0]), n_match)

            row_idx, col_idx = linear_sum_assignment(distances)

            cg_idx = [[] for _ in range(distances.shape[0])]
            for i, cp_idx in zip(row_idx, col_idx):
                cg_idx[t_idx[i]].append(cp_idx)

        elif selection_method == "minimize_loadshape_distance":
            coef_sum = n_match*len(ls_t)
            ls_t_mean = np.mean(ls_t.values, axis=0)*coef_sum

            x_opt = highs_fit_comparison_group_loadshape(
                ls_t_mean, ls_cp.values, coef_sum=coef_sum, solver="highs", settings=None, verbose=False
            )

            # argsort x_opt
            x_opt_idx = np.argsort(x_opt)[::-1][:coef_sum]

            # reshape distances to be ls_t.shape[0] x n_match
            cg_idx = np.reshape(x_opt_idx, (ls_t.shape[0], n_match))

        else:
            raise DistanceMatchingError(f"Invalid selection method: {selection_method}")

        return cg_idx
    
    
    def get_comparison_group(
        self,
        treatment_group,
        comparison_pool,
        weights=None,
    ):
        ls_t = treatment_group
        ls_cp = comparison_pool

        n_match = self.settings.N_MATCHES_PER_TREATMENT
        max_distance_threshold = self.settings.MAX_DISTANCE_THRESHOLD
        n_meters_per_chunk = self.settings.N_TREATMENTS_PER_CHUNK

        # TODO: if matching loadshapes, this isn't necessary
        distances = _distances(ls_t, ls_cp, weights, self.dist_metric, n_meters_per_chunk)

        if self.settings.ALLOW_DUPLICATE_MATCHES:
            cg_idx = self._closest_idx_duplicates_allowed(distances, n_match=n_match)
        else:
            cg_idx = self._closest_idx_duplicates_not_allowed(ls_t, ls_cp, distances)

        data = []
        for t_idx in range(ls_t.shape[0]):
            t_id = ls_t.index[t_idx]
            for cp_idx in cg_idx[t_idx]:
                cg_id = ls_cp.index[cp_idx]

                data.append([cg_id, t_id, distances[t_idx, cp_idx]])

        df = pd.DataFrame(data, columns=["id", "treatment", "distance"])

        # check that the distance is less than the threshold
        if max_distance_threshold is not None:
            df = df[df["distance"] <= max_distance_threshold]

        # add column if id is duplicated
        df["duplicated"] = df.duplicated(subset="id", keep=False)
        
        return df


class DistanceMatchingLegacy:
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
        settings=None,
    ):
        if settings is None:
            self.settings = Settings()
        elif isinstance(settings, Settings):
            self.settings = settings
        else:
            raise Exception(
                "invalid settings provided to 'individual_metering_matching'"
            )

        self.dist_metric = settings.DISTANCE_METRIC
        if self.dist_metric == "manhattan":
            self.dist_metric = "cityblock"

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
        best_match = pd.concat([best_match, next_best_match])

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
        treatment_matches_df = pd.concat([bm, tm]).sort_index()

        return treatment_matches_df

    def get_comparison_group(
        self,
        treatment_group,
        comparison_pool,
        weights=None,
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
        settings = self.settings

        # chunk the treatment group due to memory constraints
        n_treatments_per_chunk = settings.N_TREATMENTS_PER_CHUNK

        # set n_duplicate_check to be size of comparison pool
        if not settings.ALLOW_DUPLICATE_MATCHES:
            n_duplicate_check = len(comparison_pool)
        else:
            n_duplicate_check = 0

        # if you're using weights make sure to normalize the data first
        if weights:
            treatment_group = treatment_group * weights
            comparison_pool = comparison_pool * weights

        # get chunks
        treatment_group_chunks = [
            treatment_group[
                chunk * n_treatments_per_chunk : (chunk + 1) * n_treatments_per_chunk
            ]
            for chunk in range(int(len(treatment_group) / n_treatments_per_chunk) + 1)
        ]

        # for each chunk, for each of n_matches, compose a comparison group
        comparison_group = pd.DataFrame(columns=["match", "distance", "duplicated"])
        for treatment_group_chunk in treatment_group_chunks:
            mat = cdist(
                treatment_group_chunk.values,
                comparison_pool.values,
                metric=self.dist_metric,
            )
            dist_df = pd.DataFrame(mat)
            # get the best n matches
            for _ in range(settings.N_MATCHES_PER_TREATMENT):
                dist_df = dist_df[
                    dist_df.columns[~dist_df.columns.isin(comparison_group["match"])]
                ]
                if dist_df.empty:
                    continue
                new_df = self._get_best_match(dist_df, n_duplicate_check)

                if comparison_group.empty:
                    comparison_group = new_df
                else:
                    comparison_group = pd.concat([comparison_group, new_df])

        # rename columns and reindex to get original ids back
        comparison_group = comparison_group.reset_index().rename(
            {"index": "treatment"}, axis=1
        )
        comparison_group.index = comparison_pool.iloc[comparison_group["match"]].index
        comparison_group["treatment"] = treatment_group.iloc[
            comparison_group["treatment"]
        ].index
        comparison_group.drop("match", axis=1, inplace=True)
        comparison_group.index.name = "id"

        if isinstance(settings.MAX_DISTANCE_THRESHOLD, (int, float)):
            comparison_group = comparison_group[
                comparison_group["distance"] < settings.MAX_DISTANCE_THRESHOLD
            ]

        # if any duplicated, remove the duplicate with the smallest distance
        if not settings.ALLOW_DUPLICATE_MATCHES and comparison_group["duplicated"].any():
            comparison_group = comparison_group.sort_values("distance")
            comparison_group = comparison_group[~comparison_group.duplicated(subset="treatment", keep="first")]

        return comparison_group


if __name__ == "__main__":
    d = DistanceMatching()
    print(d.settings)
