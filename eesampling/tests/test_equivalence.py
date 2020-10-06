import pandas as pd

from eesampling.equivalence import *

import pytest

@pytest.fixture
def equiv_data():
    data = {
    0: [0, 0, 0],
    1: [0, 0, 1],
    2: [0, 1, 0],
    3: [0, 1, 1],
    4: [1, 0, 0],
    5: [1, 0, 1],
    6: [1, 1, 0],
    7: [1, 1, 1]
    }

    df = pd.DataFrame(data).transpose()
    return df


def equiv_data_for_binning():
    data = {
    0: [0, 0, 0],
    1: [1, 1, 1],
    2: [2, 2, 2],
    3: [3, 3, 3],
    4: [4, 4, 4],
    5: [5, 5, 5],
    6: [6, 6, 6],
    7: [7, 7, 7]
    }

    df = pd.DataFrame(data).transpose()
    return df

def test_get_quantiles():
    assert (get_quantile_indexes(1) == [0, 1]).all()
    assert (get_quantile_indexes(2) == [0, 0.5, 1]).all()
    assert (get_quantile_indexes(3) == [0, 1/3, 2/3, 1]).all()




def test_quantile_means_array():
    x = [0,1,2,3,4,5]
    assert (quantile_means_array(x, n_quantiles=1) == np.mean(x)).all()
    assert (quantile_means_array(x, n_quantiles=len(x)) == x).all()

    x = [1,1,1,10,10,10]
    assert (quantile_means_array(x, n_quantiles=2) == [1, 10]).all()


    # test assuming a 3-column array
    X = np.array(
            [[1,1,1,10,10,10],
            [1,1,1,10,10,10],
            [1,1,1,10,10,10]])

    Y = np.array(
            [[0,1,2,3,4,5],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0],
            ])

    means_x, means_y = quantile_means_population(X, Y, 2)
    assert (means_x == np.array([[1,10], [1,10], [1,10]])).all()
    assert (means_y == np.array([[1,4], [0,0], [0,0]])).all()


def test_quantile_distance():
    X = np.array(
            [[1,1,1,10,10,10],
            [1,1,1,10,10,10],
            [1,1,1,10,10,10]])

    Y = np.array(
            [[0,1,2,3,4,5],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0],
            ])

    assert (quantile_distance(X, Y, 2, "euclidean") == np.array([
            6,
            101**0.5, 
            101**0.5
        ])).all()

    assert (quantile_distance(X, Y, 2, "chisquare") == np.array([
            36/14,
            11,
            11
        ])).all()



# def test_equivalence_inputs(equiv_data):
#     ix_x = [0,1,2]
#     ix_y = [5,6,7]

#     eq = Equivalence(ix_x = [1,2], ix_y = [3 ,4], features_matrix = equiv_data)
#     assert (eq.X_x == [[0,0,1], [0,1,0]]).all()
#     assert (eq.X_y == [[0,1,1], [1,0,0]]).all()


# def test_equivalence_mean_based_euclidean(equiv_data):
#     # sqrt( sum[(x_i - y_i)^2]) 
#     eq = EquivalenceMeanBasedEuclidean(ix_x = [1,2], ix_y = [3 ,4], features_matrix = equiv_data)
#     means_x = [0, 0.5, 0.5]
#     means_y = [0.5, 0.5, 0.5]
#     distance = 0.5
#     assert eq.distance == 0.5


#     eq = EquivalenceMeanBasedEuclidean(ix_x = [0], ix_y = [7], features_matrix = equiv_data)
#     distance = 3**0.5
#     assert eq.distance == distance


# def test_equivalence_mean_based_chisquare(equiv_data):
#     # sum[ (x_i - y_i)^2 / (x_i + y_i)]
#     eq = EquivalenceMeanBasedChisquare(ix_x = [1,2], ix_y = [3 ,4], features_matrix = equiv_data)
#     means_x = [0, 0.5, 0.5]
#     means_y = [0.5, 0.5, 0.5]
#     distance = 0.5**2/0.5 
#     assert eq.distance == distance

#     eq = EquivalenceMeanBasedChisquare(ix_x = [0], ix_y = [7], features_matrix = equiv_data)
#     distance = 3
#     assert eq.distance == distance


# def test_binning(equiv_data_for_binning):
#     EquivalenceBinBased(ix_x=[0,1,2,3], ix_y=[0,1,2,3,4,5,6,7], features_matrix = equiv_data_for_binning, n_bins=1)
#     