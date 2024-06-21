"""
contains constants used for GRIDmeter
"""
from __future__ import annotations

from enum import Enum


class DistanceMetric(str, Enum):
    EUCLIDEAN = "euclidean"
    CHISQUARE = "chisquare"