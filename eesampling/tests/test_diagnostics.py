import pandas as pd

import pytest

from eesampling.diagnostics import Diagnostics
from eesampling.model import StratifiedSampling


@pytest.fixture
def diagnostics_obj(df_train, df_test, col_name):
    stratified_sampling_obj = StratifiedSampling()
    stratified_sampling_obj.add_column(col_name, n_bins=4)
    stratified_sampling_obj.fit_and_sample(
        df_train, df_test, n_samples_approx=len(df_train), random_seed=1
    )
    return stratified_sampling_obj.diagnostics()


def test_equivalence(diagnostics_obj):
    equivalence = diagnostics_obj.equivalence()
    assert equivalence["ks_ok"].all() == True and equivalence["t_ok"].all() == True


def test_record_based_equivalence_euclidean(diagnostics_obj, df_train, df_test, df_equiv):

    
    equivalence = diagnostics_obj.records_based_equivalence_euclidean(
        df_equiv, groupby_col="month", value_col="baseline_predicted_usage"
    )
    assert equivalence


def test_record_based_equivalence_chisquare(diagnostics_obj, df_train, df_test, df_equiv):

    
    equivalence = diagnostics_obj.records_based_equivalence_chisquare(
        df_equiv, groupby_col="month", value_col="baseline_predicted_usage"
    )
    assert equivalence
