import pandas as pd
import numpy as np


def get_prob_bins(df, df_comparison, col, bins=20, fixed_count=True):
    # we only care about the range within the treatment group
    min_dat = df[col].min()
    max_dat = df[col].max()
    # do it for the treatment
    if fixed_count:
        cuts, bins = pd.qcut(df[col], q=bins, duplicates="drop", retbins=True)
    else:
        bins = np.linspace(min_dat, max_dat, bins)
        cuts = pd.cut(df[col], bins=bins, include_lowest=True)
    vals = cuts.value_counts(sort=False)
    vals = vals.values / sum(vals)
    # do it for the comparison
    cuts_c = pd.cut(df_comparison[col], bins=bins, include_lowest=True)
    vals_c = cuts_c.value_counts(sort=False)
    vals_c = vals_c.values / sum(vals_c)
    return [vals, vals_c, min_dat, max_dat]


def get_kl_divs(df_treat, df_compare, **kwargs):
    # Kullback-Leibler divergence
    # bin the targeting params
    vcs = [
        get_prob_bins(df_treat, df_compare, col, **kwargs) for col in df_treat.columns
    ]
    # define the kl divergence
    def kl_(x):
        p = x[0]
        q = x[1]
        # note: q!=0 is an assumption that does not fit with the traditional
        # use of KL. It assumes that we have any lack of bins in q are due to
        # outliers in p. (where p is non zero)
        # it could be worth writing this out further
        # if there are more than 1 missing value raising more flags
        return np.sum(np.where((p != 0) & (q != 0), p * np.log(p / q), 0))

    d = pd.DataFrame(vcs, index=df_treat.columns, columns=[0, 1, "min", "max"])
    d["kl_divergence"] = d.apply(kl_, axis=1)

    return d[["kl_divergence", "min", "max"]].sort_values(
        "kl_divergence", ascending=False
    )


# choose parameters based on the correlation matrix
def choose_params(difs, corr_matrix, thresh=0.75, num_params=3):
    ordered_list = difs.index[1:]
    chosen = [difs.index[0]]
    for i in ordered_list:
        if len(chosen) == num_params:
            break
        cor = corr_matrix.loc[i]
        if any(abs(cor.loc[chosen]) > thresh):
            pass
        else:
            chosen.append(i)
    return chosen


def get_params(treatment, comparison, thresh=0.75, num_params=3, **kwargs):
    df = get_kl_divs(treatment, comparison, **kwargs)
    corr_m = treatment.corr()
    params = choose_params(df, corr_m, thresh, num_params)
    return params, df
