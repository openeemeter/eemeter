import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import chisquare


def ids_to_index(subset_ids, all_ids):
    """Convert an array of ids to an array of indexes relative to a superset of ids."""
    n = len(subset_ids)
    ix = np.ndarray(n, int)
    for i in range(n):
        index = np.where(all_ids == subset_ids[i])
        if len(index[0]) == 0:
            raise ValueError(f"ID {subset_ids[i]} not found in superset of IDs")
        ix[i] = index[0][0].astype(int)
    return ix 


# current todo:
# DONE -- outputs of equivalence computation into data frame --see rehape_outputs below
# DONE -- fix equivalence tests with new formats
# DONE -- get bin selector tests running properly
# refactor bin selector inputs, tests, notebook with new input format
# then done?  test upstream as well, in comparison-groups


class Equivalence:
    def __init__(self, ix_x, ix_y, features_matrix, n_quantiles=1, how='euclidean'):
        self.ix_x = ix_x
        self.ix_y = ix_y
        self.n_quantiles = n_quantiles
        self.how = how

        if type(features_matrix) == pd.DataFrame:
            features_matrix = features_matrix.to_numpy() # pragma: no cover
        elif type(features_matrix) == np.ndarray:
            pass
        else:
            raise ValueError("features_matrix must be a pandas DataFrame or numpy ndarray.") # pragma: no cover

        self.features_matrix = features_matrix
        self.X = self.features_matrix[ix_x].transpose()
        self.Y = self.features_matrix[ix_y].transpose()


    def compute(self):
        means_x, means_y, quantiles_x, quantiles_y = quantile_means_population(self.X, self.Y, self.n_quantiles)
        distance = sum_column_distance(means_x, means_y, how=self.how)
        equiv_x = reshape_outputs(means_x, quantiles_x)
        equiv_y = reshape_outputs(means_y, quantiles_y)
        return equiv_x, equiv_y, distance



def reshape_outputs(means, quantiles):
    out = []
    for feature in range(len(means)):
        for q in range(len(quantiles[0])-1):
            bin_label = f"[{quantiles[feature][q]}, {quantiles[feature][q+1]}]" 
            mean = means[feature][q]
            out.append({'_bin_label': bin_label, 'value': mean, 'feature_index': feature})
    return pd.DataFrame(out)



def get_quantile_indexes(n_quantiles):
    return np.linspace(0, 1, n_quantiles + 1)

def get_quantiles(col, n_quantiles):
    return np.quantile(col, get_quantile_indexes(n_quantiles))

def cut_column(col, q_this, q_next):
    # return slice of an array between q_this and q_next inclusive
    # inclusive means we include 0th and 100th percentiles, at 
    # the expense of possible duplication of middle quantiles 
    return col[(col >= q_this) & (col <= q_next)]

def quantile_means_array(col, n_quantiles):
    # slice an array into n quantiles and compute the mean value for each 
    if type(col) != np.ndarray:
        col = np.array(col)
    quantiles = get_quantiles(col, n_quantiles)
    means = np.ndarray(n_quantiles)
    for i in range(len(quantiles) - 1):
        means[i] = np.mean(cut_column(col, quantiles[i], quantiles[i+1]))
    return means, quantiles


def quantile_means_population(X, Y, n_quantiles):
    # compute means per quantile, for each column in X and Y
    n_cols = len(X)
    if not len(X) == len(Y):
        raise ValueError("Matrices must have the same number of columns.") # pragma: no cover

    means_x = np.ndarray((n_cols, n_quantiles))
    means_y = np.ndarray((n_cols, n_quantiles))
    quantiles_x = np.ndarray((n_cols, n_quantiles + 1))
    quantiles_y = np.ndarray((n_cols, n_quantiles + 1))

    for i in range(n_cols):
        col_x = X[i]
        col_y = Y[i]
        means_x[i], quantiles_x[i] = quantile_means_array(col_x, n_quantiles)
        means_y[i], quantiles_y[i] = quantile_means_array(col_y, n_quantiles)

    return means_x, means_y, quantiles_x, quantiles_y

def chisquare_dist(X,Y):
    distance = 0
    for i in range(len(X)):
        distance = distance + ((X[i] - Y[i])**2 / (X[i] + Y[i]))
    return distance 

def get_distance_func(how="euclidean"):
    if how == "euclidean":
        return lambda x, y: pdist([x,y]) 
    elif how == "chisquare":
        return chisquare_dist 
    else:
        raise ValueError(f"Unsupported distance metric: {how}") # pragma: no cover


def sum_column_distance(means_x, means_y, how="euclidean"):
    column_distances = np.ndarray(len(means_x))
    distance_func = get_distance_func(how)
    for i in range(len(means_x)):
        column_distances[i] = distance_func(means_x[i], means_y[i])
    return np.sum(column_distances)










 # def records_based_equivalence(
 #        self,
 #        df_for_equivalence,
 #        equiv_groupby_col,
 #        equiv_value_col,
 #        how,
 #        equiv_id_col="id",
 #        id_col="id",
 #        equiv_label_x=None,
 #        equiv_label_y=None,
 #        *args,
 #        **kwargs,
 #    ):
 #        if how == "euclidean":
 #            return self.records_based_equivalence_euclidean(
 #                df_for_equivalence=df_for_equivalence,
 #                equiv_groupby_col=equiv_groupby_col,
 #                equiv_value_col=equiv_value_col,
 #                equiv_id_col=equiv_id_col,
 #                id_col=id_col,
 #                equiv_label_x=equiv_label_x,
 #                equiv_label_y=equiv_label_y,
 #            )
 #        elif how == "chisquare":
 #            return self.records_based_equivalence_chisquare(
 #                df_for_equivalence=df_for_equivalence,
 #                equiv_groupby_col=equiv_groupby_col,
 #                equiv_value_col=equiv_value_col,
 #                equiv_id_col=equiv_id_col,
 #                id_col=id_col,
 #                equiv_label_x=equiv_label_x,
 #                equiv_label_y=equiv_label_y,
 #                *args,
 #                **kwargs,
 #            )
 #        else:
 #            raise ValueError("how must be one of: ['euclidean', 'chisquare']")

 #    def records_based_equivalence_euclidean(
 #        self,
 #        df_for_equivalence,
 #        equiv_groupby_col,
 #        equiv_value_col,
 #        equiv_id_col="id",
 #        id_col="id",
 #        equiv_label_x=None,
 #        equiv_label_y=None,
 #    ):
 #        """
 #        Attributes
 #        ----------
 #        equiv_label_x: str
 #            First label to measure equivalence against (defaults to treatment label) 
 #        equiv_label_y: str
 #            Second label to measure equivalence against (defaults to sample if available,
 #             otherwise defaults to full pool set)
 #        """
 #        equiv_label_x, equiv_label_y = self._check_equiv_labels(
 #            equiv_label_x, equiv_label_y
 #        )

 #        df = self.df_all[["population", id_col]].copy()
 #        df_combined = df.set_index(id_col, drop=True).join(
 #            df_for_equivalence.set_index(equiv_id_col, drop=True)
 #        )
 #        equiv_x = (
 #            df_combined[df_combined["population"] == equiv_label_x]
 #            .groupby(equiv_groupby_col)[equiv_value_col]
 #            .mean()
 #        )
 #        equiv_y = (
 #            df_combined[df_combined["population"] == equiv_label_y]
 #            .groupby(equiv_groupby_col)[equiv_value_col]
 #            .mean()
 #        )
 #        return equiv_x.to_frame(), equiv_y.to_frame(), pdist([equiv_x, equiv_y])[0]


 #    def records_based_equivalence_chisquare(
 #        self,
 #        df_for_equivalence,
 #        equiv_groupby_col,
 #        equiv_value_col,
 #        equiv_id_col="id",
 #        id_col="id",
 #        equiv_label_x=None,
 #        equiv_label_y=None,
 #        chisquare_n_values_per_bin=25,
 #        chisquare_is_fixed_width=False,
 #    ):
 #        """
 #        Attributes
 #        ----------
 #        equiv_label_x: str
 #            First label to measure equivalence against (defaults to treatment label) 
 #        equiv_label_y: str
 #            Second label to measure equivalence against (defaults to sample if available,
 #             otherwise defaults to full pool set)
 #        """
 #        equiv_label_x, equiv_label_y = self._check_equiv_labels(
 #            equiv_label_x, equiv_label_y
 #        )

 #        df = self.df_all[["population", id_col]].copy()
 #        df_combined = df.set_index(id_col, drop=True).join(
 #            df_for_equivalence.set_index(equiv_id_col, drop=True)
 #        )
 #        features = df_combined[equiv_groupby_col].unique()
 #        df_combined = df_combined.set_index(['population', equiv_groupby_col]).sort_index()

 #        chisquare_stats = []
 #        equiv_x = []
 #        equiv_y = []        
 #        for groupby_col_num in features:
           
 #            df_equiv_x = df_combined.loc[equiv_label_x, groupby_col_num]
 #            df_equiv_y = df_combined.loc[equiv_label_y, groupby_col_num]             
 #            # Select num_bins from number of treatments / chosen ratio
 #            num_bins = max(int(len(df_equiv_x) / chisquare_n_values_per_bin), 1)

 #            binned_data_x = df_equiv_x
 #            binned_data_x['_bin_label'] = pd.qcut(binned_data_x[equiv_value_col], q=num_bins).astype(str)

 #            binned_data_y = df_equiv_y
 #            binned_data_y['_bin_label'] = pd.qcut(binned_data_y[equiv_value_col], q=num_bins).astype(str)

 #            chisquare_x = binned_data_x.groupby("_bin_label")[equiv_value_col].mean()
 #            chisquare_y = binned_data_y.groupby("_bin_label")[equiv_value_col].mean()
 #            chisquare_stats.append(chisquare(chisquare_x, chisquare_y).statistic)

 #            chisquare_x_df = chisquare_x.to_frame()
 #            chisquare_x_df[equiv_groupby_col] = groupby_col_num
 #            chisquare_y_df = chisquare_y.to_frame()
 #            chisquare_y_df[equiv_groupby_col] = groupby_col_num

 #            equiv_x.append(chisquare_x_df)
 #            equiv_y.append(chisquare_y_df)
 #        return (
 #            pd.concat(equiv_x).reset_index(),
 #            pd.concat(equiv_y).reset_index(),
 #            sum(chisquare_stats) if chisquare_stats else np.inf,
 #        )

 # 