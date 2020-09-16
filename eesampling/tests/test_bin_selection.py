from eesampling.model import StratifiedSampling, BinnedData
from eesampling.bin_selection import StratifiedSamplingBinSelector
from eesampling.bins import ModelSamplingException
import pytest


def test_stratified_sampling_fit_and_sample_records_equivalence(
    df_train, df_test, df_equiv, col_name
):
    stratified_sampling_obj = StratifiedSampling()
    df_test["col2"] = df_test[col_name]
    df_train["col2"] = df_train[col_name]
    stratified_sampling_obj.add_column(col_name)
    stratified_sampling_obj.add_column("col2")
    ## attempting to estimate both n_bins and n_samples
    StratifiedSamplingBinSelector(stratified_sampling_obj,
        df_train,
        df_test,
        df_equiv,
        equivalence_groupby_col="month",
        equivalence_value_col="baseline_predicted_usage",
        equivalence_id_col="id",
        min_n_bins=4,
        max_n_bins=6,
        random_seed=1,
        how='chisquare'
    )
    output = stratified_sampling_obj.data_sample.df
    bins_df = stratified_sampling_obj.diagnostics().count_bins()


def test_stratified_sampling_fit_and_sample_records_equivalence_too_many_bins(
    df_train, df_test, df_equiv, col_name
):
    stratified_sampling_obj = StratifiedSampling()

    stratified_sampling_obj.add_column(col_name)
    ## attempting to estimate both n_bins and n_samples
    with pytest.raises(ModelSamplingException):
        model_w_selected_bins = StratifiedSamplingBinSelector(stratified_sampling_obj,
            df_train,
            df_test,
            df_equiv,
            equivalence_groupby_col="month",
            equivalence_value_col="baseline_predicted_usage",
            equivalence_id_col="id",
            min_n_bins=1000,
            max_n_bins=1002,
            random_seed=1,
            how='chisquare',
            relax_n_samples_approx_constraint=False,
        )


def test_stratified_sampling_fit_and_sample_records_equivalence_idempotent_check(
    df_train, df_test, df_equiv, col_name
):
    df_train["col2"] = df_train[col_name] * 2
    df_train["col3"] = df_train[col_name] * 3

    df_test["col2"] = df_test[col_name] * 2
    df_test["col3"] = df_test[col_name] * 3

    stratified_sampling_obj = StratifiedSampling()
    stratified_sampling_obj.add_column(col_name)
    stratified_sampling_obj.add_column("col2")
    stratified_sampling_obj.add_column("col3")

    StratifiedSamplingBinSelector(stratified_sampling_obj,
        df_train,
        df_test,
        df_equiv,
        equivalence_groupby_col="month",
        equivalence_value_col="baseline_predicted_usage",
        equivalence_id_col="id",
        min_n_bins=2,
        max_n_bins=3,
        random_seed=1,
        how='chisquare'
    )
    sample1 = stratified_sampling_obj.data_sample.df.index.values

    stratified_sampling_obj = StratifiedSampling()
    stratified_sampling_obj.add_column(col_name)
    stratified_sampling_obj.add_column("col2")
    stratified_sampling_obj.add_column("col3")
    StratifiedSamplingBinSelector(stratified_sampling_obj,
        df_train,
        df_test,
        df_equiv,
        equivalence_groupby_col="month",
        equivalence_value_col="baseline_predicted_usage",
        equivalence_id_col="id",
        min_n_bins=2,
        max_n_bins=3,
        random_seed=1,
        how='chisquare'
    )
    sample2 = stratified_sampling_obj.data_sample.df.index.values
    assert set(sample1) == set(sample2)


def test_stratified_sampling_fit_and_sample_records_equivalence_euclidean_idempotent_check(
    df_train, df_test, df_equiv, col_name
):
    df_train["col2"] = df_train[col_name] * 2
    df_train["col3"] = df_train[col_name] * 3

    df_test["col2"] = df_test[col_name] * 2
    df_test["col3"] = df_test[col_name] * 3

    stratified_sampling_obj = StratifiedSampling()
    stratified_sampling_obj.add_column(col_name)
    stratified_sampling_obj.add_column("col2")
    stratified_sampling_obj.add_column("col3")

    StratifiedSamplingBinSelector(stratified_sampling_obj,
        df_train,
        df_test,
        df_equiv,
        equivalence_groupby_col="month",
        equivalence_value_col="baseline_predicted_usage",
        equivalence_id_col="id",
        min_n_bins=2,
        max_n_bins=3,
        random_seed=1,
        how='euclidean'
    )
    sample1 = stratified_sampling_obj.data_sample.df.index.values

    stratified_sampling_obj = StratifiedSampling()
    stratified_sampling_obj.add_column(col_name)
    stratified_sampling_obj.add_column("col2")
    stratified_sampling_obj.add_column("col3")
    StratifiedSamplingBinSelector(stratified_sampling_obj,
        df_train,
        df_test,
        df_equiv,
        equivalence_groupby_col="month",
        equivalence_value_col="baseline_predicted_usage",
        equivalence_id_col="id",
        min_n_bins=2,
        max_n_bins=3,
        random_seed=1,
        how='euclidean'
    )
    sample2 = stratified_sampling_obj.data_sample.df.index.values
    assert set(sample1) == set(sample2)


def test_stratified_sampling_fit_and_sample_records_equivalence_euclidean_idempotent_check(
    df_train, df_test, df_equiv, col_name
):
    df_train["col2"] = df_train[col_name] * 2
    df_train["col3"] = df_train[col_name] * 3

    df_test["col2"] = df_test[col_name] * 2
    df_test["col3"] = df_test[col_name] * 3

    stratified_sampling_obj = StratifiedSampling()
    stratified_sampling_obj.add_column(col_name)
    stratified_sampling_obj.add_column("col2")
    stratified_sampling_obj.add_column("col3")

    StratifiedSamplingBinSelector(stratified_sampling_obj,
        df_train,
        df_test,
        df_equiv,
        equivalence_groupby_col="month",
        equivalence_value_col="baseline_predicted_usage",
        equivalence_id_col="id",
        min_n_bins=2,
        max_n_bins=3,
        random_seed=1,
        how='euclidean'
    )
    sample1 = stratified_sampling_obj.data_sample.df.index.values

    stratified_sampling_obj = StratifiedSampling()
    stratified_sampling_obj.add_column(col_name)
    stratified_sampling_obj.add_column("col2")
    stratified_sampling_obj.add_column("col3")
    StratifiedSamplingBinSelector(stratified_sampling_obj,
        df_train,
        df_test,
        df_equiv,
        equivalence_groupby_col="month",
        equivalence_value_col="baseline_predicted_usage",
        equivalence_id_col="id",
        min_n_bins=2,
        max_n_bins=3,
        random_seed=1,
        how='euclidean'
    )
    sample2 = stratified_sampling_obj.data_sample.df.index.values
    assert set(sample1) == set(sample2)


def test_plot_records_based_equiv_average(
    df_train, df_test, df_equiv, col_name
):
    df_train["col2"] = df_train[col_name] * 2
    df_train["col3"] = df_train[col_name] * 3

    df_test["col2"] = df_test[col_name] * 2
    df_test["col3"] = df_test[col_name] * 3

    stratified_sampling_obj = StratifiedSampling()
    stratified_sampling_obj.add_column(col_name)
    stratified_sampling_obj.add_column("col2")
    stratified_sampling_obj.add_column("col3")

    bin_selection = StratifiedSamplingBinSelector(stratified_sampling_obj,
        df_train,
        df_test,
        df_equiv,
        equivalence_groupby_col="month",
        equivalence_value_col="baseline_predicted_usage",
        equivalence_id_col="id",
        min_n_bins=2,
        max_n_bins=3,
        random_seed=1,
        how='euclidean'
    )
    bin_selection.plot_records_based_equiv_average(plot=False)
    bin_selection.results_as_json()


def test_plot_records_based_equiv_average_chisquare(
    df_train, df_test, df_equiv, col_name
):
    df_train["col2"] = df_train[col_name] * 2
    df_train["col3"] = df_train[col_name] * 3

    df_test["col2"] = df_test[col_name] * 2
    df_test["col3"] = df_test[col_name] * 3

    stratified_sampling_obj = StratifiedSampling()
    stratified_sampling_obj.add_column(col_name)
    stratified_sampling_obj.add_column("col2")
    stratified_sampling_obj.add_column("col3")

    bin_selection = StratifiedSamplingBinSelector(stratified_sampling_obj,
        df_train,
        df_test,
        df_equiv,
        equivalence_groupby_col="month",
        equivalence_value_col="baseline_predicted_usage",
        equivalence_id_col="id",
        min_n_bins=2,
        max_n_bins=3,
        random_seed=1,
        how='chisquare'
    )
    bin_selection.plot_records_based_equiv_average(plot=False)
    results = bin_selection.results_as_json()
    assert 'bins_selected_str' in list(results['n_bin_results'][0].keys())
