import pandas as pd

from eesampling.equivalence import *

import pytest
import numpy as np


@pytest.fixture
def equiv_X():
    # 3 columns, 6 rows, column first 
    return np.array(
            [[1,1,1,10,10,10],
            [1,1,1,10,10,10],
            [1,1,1,10,10,10]])


@pytest.fixture
def equiv_Y():
    # 3 columns, 6 rows, column first 
    return np.array(
            [[0,1,2,3,4,5],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0],
            ])


@pytest.fixture
def feature_matrix(equiv_X, equiv_Y):
    x = equiv_X
    y = equiv_Y
    return np.concatenate([x.transpose(), y.transpose()])


def test_get_quantiles():
    assert (get_quantile_indexes(1) == [0, 1]).all()
    assert (get_quantile_indexes(2) == [0, 0.5, 1]).all()
    assert (get_quantile_indexes(3) == [0, 1/3, 2/3, 1]).all()




def test_quantile_means_array(equiv_X, equiv_Y):
    x = [0,1,2,3,4,5]
    assert (quantile_means_array(x, n_quantiles=1) == np.mean(x)).all()
    assert (quantile_means_array(x, n_quantiles=len(x)) == x).all()

    x = [1,1,1,10,10,10]
    assert (quantile_means_array(x, n_quantiles=2) == [1, 10]).all()

    means_x, means_y = quantile_means_population(equiv_X, equiv_Y, 2)
    assert (means_x == np.array([[1,10], [1,10], [1,10]])).all()
    assert (means_y == np.array([[1,4], [0,0], [0,0]])).all()


def test_quantile_distance(equiv_X, equiv_Y):
    X = equiv_X
    Y = equiv_Y

    means_x, means_y, distance, column_distances = quantile_distance(X, Y, 2, "euclidean")
    assert round(distance, 2) == round(6 + 2*101**0.5, 2)
    assert (column_distances == np.array([
            6,
            101**0.5, 
            101**0.5
        ])).all()

    means_x, means_y, distance, column_distances = quantile_distance(X, Y, 2, "chisquare")
    assert round(distance,2) == round(36/14 + 2*11, 2)
    assert (column_distances == np.array([
            36/14,
            11,
            11
        ])).all()



def test_equivalence_inputs(feature_matrix):

    eq = Equivalence(ix_x = [1,2], ix_y = [3, 10, 11], features_matrix = feature_matrix)
    
    # X should be [1,1,1] and [1,1,1]
    # eq.X should be sliced by column so [1,1], [1,1], [1,1]
    # Y should be [10, 10, 10], [4, 0, 0], [5, 0, 0]
    # eq.Y should be [10, 4, 5], [10, 0, 0], [10, 0 0]
    assert (eq.X == np.array([[1,1], [1,1], [1,1]])).all()
    assert (eq.Y == np.array([[10,4,5], [10,0,0], [10,0,0]])).all()


def test_equivalene_distance(feature_matrix):

    eq = Equivalence(ix_x = [0,1,2,3,4,5], ix_y = [6,7,8,9,10,11], 
        features_matrix = feature_matrix, n_quantiles=2, how="euclidean")
    means_x, means_y, distance, column_distances = eq.compute()

    assert (means_x == np.array([[1,10], [1,10], [1,10]])).all()
    assert (means_y == np.array([[1,4], [0,0], [0,0]])).all()
    assert round(distance,2) == round(6 + 2*101**0.5,2)
    assert (column_distances == np.array([
            6,
            101**0.5, 
            101**0.5
        ])).all()


    eq = Equivalence(ix_x = [0,1,2,3,4,5], ix_y = [6,7,8,9,10,11], 
        features_matrix = feature_matrix, n_quantiles=2, how="chisquare")
    means_x, means_y, distance, column_distances = eq.compute()

    assert (means_x == np.array([[1,10], [1,10], [1,10]])).all()
    assert (means_y == np.array([[1,4], [0,0], [0,0]])).all()
    assert round(distance,2) == round(36/14 + 2*11,2)
    assert (column_distances == np.array([
            36/14,
            11, 
            11
        ])).all()

