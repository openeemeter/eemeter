"""
contains constants used for GRIDmeter
"""
from __future__ import annotations

from enum import Enum


class DistanceMetric(str, Enum):
    """
    what distance method to use
    """

    EUCLIDEAN = "euclidean"
    SEUCLIDEAN = "seuclidean"
    MANHATTAN = "manhattan"
    COSINE = "cosine"


class AggType(str, Enum):
    MEAN = "mean"
    MEDIAN = "median"
