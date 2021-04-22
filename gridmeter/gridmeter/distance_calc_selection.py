import numpy as np
import pandas as pd
import scipy


class DistanceMatching:
    def __init__(
        self,
        treatment_group,
        comparison_pool,
        weights=None,
        n_treatments_per_chunk=10000,
    ):
        """
        Attributes
        ----------
        treatment_group: pd.DataFrame
            A
        n_matches_per_treatment: int
            number of comparison matches desired per treatment

        """
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
        return pd.Series(dist_df.columns[(np.argmin(dist_df.values, axis=1))])

    def _get_next_best_matches(
        self, treatment_distances_df, treatment_matches, treatment_matches_duplicated
    ):
        # The purpose of this for loop is to attempt to find the 'next best match'
        # for treatment meters matched to a comparison pool meter that has already
        # had a previous match

        # Get a matrix that only contains the treatments
        # that were matched to an already matched comparison pool meter
        # and therefore need a 'next-best' match
        treatment_distances_unmatched_df = treatment_distances_df.loc[
            treatment_distances_df.index[treatment_matches_duplicated.index]
        ]

        # Remove the columns from this matrix
        # that refer to comparison pool meters that have already been matched
        treatment_distances_unmatched_df.drop(
            treatment_matches.unique(), axis=1, inplace=True
        )
        if treatment_distances_unmatched_df.columns.empty:
            # There are no unmatched comparison pool meters remaining
            return treatment_matches, treatment_matches_duplicated, False

        # Next-best match is found for the unmatched treatment meters
        treatment_matches_next_best = self._get_min_distance_from_matrix_df(
            treatment_distances_unmatched_df
        )

        # Assign these 'next-best' matches to our full list
        treatment_matches[
            treatment_distances_unmatched_df.index
        ] = treatment_matches_next_best.values

        # Check if there are still duplicates remaining and stop if there are none
        treatment_matches_duplicated = treatment_matches[treatment_matches.duplicated()]
        if treatment_matches_duplicated.empty:
            return treatment_matches, treatment_matches_duplicated, False
        return treatment_matches, treatment_matches_duplicated, True

    def _get_best_match(self, treatment_distances_df, n_max_duplicate_check_rounds):
        """
        Attributes
        ----------
        treatment_distances_df: pd.DataFrame
            A matrix where the row indices (i) are treatment meters, the columns (j) are
            comparison pool meters, and the values are the calculated distance between
            treatment[i] and comparison_pool[j]
        """

        treatment_matches = self._get_min_distance_from_matrix_df(
            treatment_distances_df
        )
        treatment_matches_duplicated = treatment_matches[treatment_matches.duplicated()]

        for run_i in range(0, n_max_duplicate_check_rounds):
            (
                treatment_matches,
                treatment_matches_duplicated,
                check_again,
            ) = self._get_next_best_matches(
                treatment_distances_df, treatment_matches, treatment_matches_duplicated
            )
            if not check_again:
                break

        treatment_matches_df = treatment_matches.to_frame(name="match")

        treatment_matches_df["distance"] = treatment_matches_df.apply(
            lambda row: treatment_distances_df.loc[row.name, row["match"]], axis=1
        )

        return treatment_matches_df

    def get_comparison_group(
        self,
        n_matches_per_treatment,
        metric="euclidean",
        max_distance_threshold=None,
        n_max_duplicate_check_rounds=10,
    ):
        """
        n_matches_per_treatment: int
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
                new_df = self._get_best_match(dist_df, n_max_duplicate_check_rounds)
                comparison_group = comparison_group.append(new_df)

        # Label any remaining duplicates
        comparison_group["duplicated"] = comparison_group["match"].apply(
            lambda x: x
            in comparison_group[comparison_group["match"].duplicated()][
                "match"
            ].unique()
        )

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
