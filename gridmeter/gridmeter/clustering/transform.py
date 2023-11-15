"""
module which contains (for now) both utilities
for clustering transforms and the transform logic itself
"""

from __future__ import annotations

import attrs
import numpy as np
import pandas as pd

import skfda
import skfda.representation.grid
import skfda.representation.basis
import skfda.preprocessing.dim_reduction.feature_extraction

from typing import cast


_NORMALIZATION_QUANTILE = 0.1


def unstack_and_ensure_df(df: pd.DataFrame):
    """
    util function which unstacks a DataFrame and ensures the result is still a DataFrame.

    Important because some can become Series which will cause issues in the code.

    Also helps with type checking by always returning a DataFrame
    """

    unstack = df.unstack()
    if not isinstance(unstack, pd.DataFrame):
        raise ValueError(
            f"attempted to unstack a DataFrame and resulted in type not being a DataFrame when explicitly set to be one: type - {type(unstack)}"
        )

    return unstack


def _stack_and_ensure_df(df: pd.DataFrame):
    """
    util function which stacks a DataFrame and ensures the result is still a DataFrame.

    Important because some can become Series which will cause issues in the code.

    Also helps with type checking by always returning a DataFrame
    """

    stacked = df.stack()
    if not isinstance(stacked, pd.DataFrame):
        raise ValueError(
            f"attempted to stack a DataFrame and resulted in type not being a DataFrame when explicitly set to be one: type - {type(stacked)}"
        )

    return stacked


def _drop_nonfinite_from_df(df: pd.DataFrame):
    """
    also ensures that dropping nonfinite result is still a dataframe.

    can be a series if not multi-index and will cause issues if so.
    """
    df = df.replace([np.inf, -np.inf], np.nan)
    # with pd.option_context("mode.use_inf_as_null", True):
    unstacked = unstack_and_ensure_df(df=df)
    dropped = unstacked.dropna(axis=1, how="any")
    return _stack_and_ensure_df(df=dropped)


@attrs.define
class _TransformInput:
    """
    contains the numpy array used for the various transforms needed
    for clustering.

    creating as a class so all necessary components needed for converting back to a dataframe
    are provided.
    """

    ids: pd.Index
    """ids needed to reconstruct dataframe from array"""

    ls_values: np.ndarray
    """
    the loadshape values as an array. often used as basis of transform
    """

    is_for_fpca: bool
    """
    flag for if the inputs are specifically for fpca. changes a few of the internal fields
    """

    name: str

    time_arr: np.ndarray
    """
    time_arr is always the hours unlike columns which is only sometimes.
    Needed to pass to certain transforms (fPCA)
    """

    time_idx: pd.Index

    @classmethod
    def from_multi_index_dataframe(
        cls, df: pd.DataFrame, is_for_fpca: bool, drop_nonfinite: bool
    ):
        """
        
        """
        if drop_nonfinite:
            df = _drop_nonfinite_from_df(df=df)

        df_unstack = unstack_and_ensure_df(df=df)
        ls_values = df_unstack.values

        ids = df_unstack.index

        # USED TO BE IN IF CONDITIONAL IF THE INDEX WAS A MULTI_INDEX, NOW ASSUMING IT ALWAYS IS
        time = df_unstack.columns.get_level_values("hour")

        name = "ls"

        time_arr = time.to_numpy().astype(float)

        return _TransformInput(
            ids=ids,
            ls_values=ls_values,
            is_for_fpca=is_for_fpca,
            name=name,
            time_arr=time_arr,
            time_idx=time,
        )

    def _get_columns(self, transformed_ls_values: np.ndarray):
        if not self.is_for_fpca:
            return self.time_idx

        return pd.Series(
            np.arange(transformed_ls_values.shape[1]) + 1,
            #
            name="fPCA_component",
        )

    def get_transformed_result_as_df(
        self, transformed_ls_values: np.ndarray
    ) -> pd.DataFrame:
        """
        uses saved fields and an np array that was transformed
        to create a dataframe
        """
        columns = self._get_columns(transformed_ls_values=transformed_ls_values)
        return (
            pd.DataFrame(transformed_ls_values, index=self.ids, columns=columns)
            .stack()
            .rename(self.name)  # type: ignore
        )


def _normalize_transposed_loadshape_np_arr_min_max(ls_arr: np.ndarray):
    """
    applies the min and max normalization logic to a dataframe which contains the transformed
    loadshape
    """
    ls_arr_transposed = ls_arr.T
    a, b = [0, 1]  # range to normalize to

    # current range
    ls_min, ls_max = np.quantile(
        ls_arr_transposed,
        [_NORMALIZATION_QUANTILE, 1 - _NORMALIZATION_QUANTILE],
        axis=0,
    )

    ret_arr = (b - a) * (ls_arr_transposed - ls_min) / (ls_max - ls_min) + a
    return ret_arr.T


def get_min_maxed_normalized_unstacked_ls_df(ls_df: pd.DataFrame, drop_nonfinite: bool):
    """
    receives loadshape dataframe and applies min max normalization
    on the values and returns a dataframe where the hours are the columns

    This is done before performing FPCA on the comparison pool
    """

    t_in = _TransformInput.from_multi_index_dataframe(
        df=ls_df, is_for_fpca=False, drop_nonfinite=drop_nonfinite
    )

    normalized_np_arr = _normalize_transposed_loadshape_np_arr_min_max(
        ls_arr=t_in.ls_values
    )

    df = t_in.get_transformed_result_as_df(transformed_ls_values=normalized_np_arr)
    return unstack_and_ensure_df(df=df)


def _get_all_normalized_ls_dfs_as_list(
    total_cp_df: pd.DataFrame,
) -> list[pd.DataFrame]:
    """
    iterates through unique indices to extract all dataframes to be normalized as a list
    """
    dataframes = []
    ids = total_cp_df.index.get_level_values("id").unique().values
    for _id in ids:
        data = total_cp_df.iloc[total_cp_df.index.get_level_values("id") == _id]
        dataframes.append(data)

    return dataframes


def concatenate_normalized_ls_dfs(ls_dfs: list[pd.DataFrame]):
    """
    concatenates the normalized loadshape dataframes into a single one ready for fpca
    """
    concat_df = pd.concat(ls_dfs)
    srs = concat_df.stack()
    srs = cast(pd.Series, srs)
    return srs.to_frame(name="ls")


def _get_fpca(
    x: np.ndarray, y: np.ndarray, min_var_ratio: float
) -> tuple[np.ndarray, str | None]:
    """
    applies fpca to concatenated transform loadshape dataframe values

    x -> time converted to np array taken from loadshape dataframe
    y -> transformed values

    assumes mixture_components return and fourier basis

    also may return a string as second return value. if it is not None, it implies an error occurred
    """

    if 0 >= min_var_ratio or min_var_ratio >= 1:
        raise ValueError("min_var_ratio but be greater than 0 and less than 1")

    n_min = 1
    # get maximum n components
    n_max = np.min(
        np.array(np.shape(y)) - [1, 5]
    )  # smallest 1  || min(largest = number of samples - 1, # time points)
    if n_max < n_min:
        n_max = n_min

    # get maximum principle components
    if len(x) == 0 or len(y) == 0:
        return np.array([]), "provided empty values for fpca"

    n_max = cast(int, n_max)

    fd = skfda.representation.grid.FDataGrid(grid_points=x, data_matrix=y)
    basis_fcn = skfda.representation.basis.Fourier

    basis_fd = fd.to_basis(basis_fcn(n_basis=n_max + 4))
    fpca = skfda.preprocessing.dim_reduction.feature_extraction.FPCA(
        n_components=n_max, components_basis=basis_fcn(n_basis=n_max + 4)
    )
    fpca.fit(basis_fd)

    var_ratio = np.cumsum(fpca.explained_variance_ratio_) - min_var_ratio
    n = int(np.argmin(var_ratio < 0.0) + 1)

    basis_fd = fd.to_basis(basis_fcn(n_basis=n + 4))
    fpca = skfda.preprocessing.dim_reduction.feature_extraction.FPCA(
        n_components=n, components_basis=basis_fcn(n_basis=n + 4)
    )
    fpca.fit(basis_fd)

    mixture_components = fpca.transform(basis_fd)
    return mixture_components, None


def _get_fpca_from_concatenate_normalized_ls(
    concat_df: pd.DataFrame, min_var_ratio: float
) -> tuple[pd.DataFrame, str | None]:
    """
    function which receives concatenated dataframe of normalized loadshape
    dataframes and a min_var_ratio and then performs fpca.

    Also returns a string that if not empty, implies an error message/something went wrong
    """
    t_in = _TransformInput.from_multi_index_dataframe(
        df=concat_df, is_for_fpca=True, drop_nonfinite=True
    )

    fcpa_mixture_components, err_msg = _get_fpca(
        x=t_in.time_arr, y=t_in.ls_values, min_var_ratio=min_var_ratio
    )
    if err_msg is not None:
        return pd.DataFrame(), None

    return (
        t_in.get_transformed_result_as_df(
            transformed_ls_values=fcpa_mixture_components
        ),
        None,
    )



@attrs.define
class InitialPoolLoadshapeTransform:

    """
    contains result of initial transform applied to loadshape

    contains the concatenated result of pool loadshapes as well
    as the dataframe result of fpca on these transformed loadshapes.
    """

    fpca_result: pd.DataFrame
    """
    contains the resulting dataframe where fpca has been applied to the loadshape
    """

    concatenated_loadshapes: pd.DataFrame
    """
    contains the concatenation of all loadshapes pertaining to this group.
    """

    err_msg: str | None
    """error message that if present, means something went wrong and the transform should be discarded"""
    
    @classmethod
    def from_concat_df_and_min_var_ratio(
        cls,
        concat_transform_df: pd.DataFrame,
        min_var_ratio: float,
        concat_loadshape_df: pd.DataFrame,
    ):
        """
        classmethod which creates the dataclass assumming that the concat_df provided
        is the concatenated result of all the transformed loadshapes that will have fpca performed on it
        """

        fpca_result, err_msg = _get_fpca_from_concatenate_normalized_ls(
            concat_df=concat_transform_df, min_var_ratio=min_var_ratio
        )
        return InitialPoolLoadshapeTransform(
            fpca_result=fpca_result,
            concatenated_loadshapes=concat_loadshape_df,
            err_msg=err_msg,
        )
    
    @classmethod
    def from_full_cp_ls_df(cls, df: pd.DataFrame, min_var_ratio: float):
        """
        classmethod to create dataclass with transform results

        applies normalization to each loadshape and then perform fpca so the data
        is ready for clustering
        """

        all_cp_dfs = _get_all_normalized_ls_dfs_as_list(total_cp_df=df)
        transformed_dfs = [
            get_min_maxed_normalized_unstacked_ls_df(ls_df=df, drop_nonfinite=True)
            for df in all_cp_dfs
        ]
        concat_transformed = concatenate_normalized_ls_dfs(ls_dfs=transformed_dfs)

        fpca_result, err_msg = _get_fpca_from_concatenate_normalized_ls(
            concat_df=concat_transformed, min_var_ratio=min_var_ratio
        )

        return InitialPoolLoadshapeTransform(
            fpca_result=fpca_result,
            concatenated_loadshapes=df,
            err_msg=err_msg,
        )

