import numpy as np
import pandas as pd
import scipy


def get_best_match(dist_df, max_runs_allowed):
    mins = pd.Series(dist_df.columns[(np.argmin(dist_df.values, axis=1))])
    mins_dupe = mins[mins.duplicated()]
    n = 1
    while (mins.nunique() < len(dist_df)) & (n < max_runs_allowed):
        dist_df_dupe = dist_df.loc[dist_df.index[mins_dupe.index]]
        # drop all the already selected entries
        dist_df_dupe.drop(mins.unique(), axis=1, inplace=True)
        mins_new = pd.Series(
            dist_df_dupe.columns[np.argmin(dist_df_dupe.values, axis=1)]
        )
        # get rid of the original duplicates to append later
        mins.loc[mins_dupe.index] = mins_new.values
        mins_dupe = mins[mins.duplicated()]
        n = n + 1
    dists = [dist_df.loc[dist_df.index[i], mins[i]] for i in mins.index]
    df = pd.DataFrame({"matches": mins, "distance": dists}, index=dist_df.index)
    if n == max_runs_allowed:
        print("warning did not converge")
        print("duplicates present")
    return df


def distance_match(
    treatment,
    comparison_pool,
    n_comparison_group_meters_per_treatment=4,
    limit_max_dist=False,
    max_dist_threshold=None,
    max_runs_allowed=10,
    weights=None,
    max_treatment_per_chunk=10000,
    allow_multiple_matches_per_comparison_pool_meter=True,
):
    """
    n_comparison_group_meters_per_treatment - number of comparison matches desired per treatment
    """
    weights = weights if weights else np.ones(len(treatment.columns))
    treatment = treatment * weights
    comparison_pool = comparison_pool * weights
    # compute the distance

    def _get_best_match(treatment_chunk):
        mat = scipy.spatial.distance.cdist(
            treatment_chunk.values,
            comparison_pool.values,
            metric="euclidean",
        )
        dist_df = pd.DataFrame(mat)
        df = pd.DataFrame(columns=["matches", "distance"])
        # get the best n matches
        for n in range(n_comparison_group_meters_per_treatment):
            dist_df = dist_df[dist_df.columns[~dist_df.columns.isin(df.matches)]]
            new_df = get_best_match(dist_df, max_runs_allowed)
            df = df.append(new_df)
        return df

    treatment_chunks = [
        treatment[
            chunk * max_treatment_per_chunk : (chunk + 1) * max_treatment_per_chunk
        ]
        for chunk in range(int(len(treatment) / max_treatment_per_chunk) + 1)
    ]

    dfs = []
    for treatment_chunk in treatment_chunks:
        df_chunk = _get_best_match(treatment_chunk)
        dfs.append(df_chunk)
    df = pd.concat(dfs)

    # TODO (deal with unwanted duplicate matches, sometimes its ok though)
    if not allow_multiple_matches_per_comparison_pool_meter:
        pass

    df = df.reset_index().rename({"index": "treatment"}, axis=1)
    df.index = comparison_pool.iloc[df["matches"]].index
    df["treatment"] = treatment.iloc[df["treatment"]].index
    df.drop("matches", axis=1, inplace=True)
    df.index.name = "id"

    return df[df["distance"] < max_dist_threshold] if max_dist_threshold else df
